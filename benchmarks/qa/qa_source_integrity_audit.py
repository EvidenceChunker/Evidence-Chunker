import sys, re, json, csv, hashlib, argparse, traceback, time
from collections import Counter, defaultdict
from pathlib import Path
from importlib.metadata import version

FORBIDDEN_MODULES = {'evidence_chunker','smart_chunker','interfaces','context_attacher',
  'table_splitter','token_counter','bbox_utils','langchain_wrapper','_table_utils',
  '_caption_mapper'}

def assert_isolation():
    loaded = {m.split('.')[0] for m in sys.modules} & FORBIDDEN_MODULES
    if loaded: raise RuntimeError('QA 검증 격리 위반: ' + str(sorted(loaded)))

assert_isolation()

def norm(s):
    if not s: return ''
    s = str(s).lower().replace('\xa0', ' ')
    s = re.sub(r'[^\w.\-±× ]+', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    s = re.sub(r'\.$', '', s)
    return s

def has_token(needle, haystack_norm):
    n = norm(needle)
    return bool(n and re.search(r'(?<!\d)' + re.escape(n) + r'(?!\d)', haystack_norm))

def em_hit(spec, text):
    c = norm(text)
    return (has_token(spec.get('value',''), c)
            and all(has_token(k, c) for k in spec.get('context_keys', [])))

def has_keys(spec,text):
    c = norm(text)
    return all(has_token(k,c) for k in spec.get('context_keys',[]))

def sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
    return h.hexdigest()

def table_md(doc,t):
    for fn in (lambda:t.export_to_markdown(doc),lambda:t.export_to_markdown()):
        try:
            m = fn()
            if m: return str(m)
        except Exception: pass
    return ''

def table_items(doc):
    out=[]
    for i,t in enumerate(getattr(doc,'tables',[]) or []):
        p=(t.prov or [])
        pg=p[0].page_no if p else None
        data=getattr(t,'data',None)
        out.append({'index':i,'page':pg,'markdown':table_md(doc,t),
          'rows':getattr(data,'num_rows',None),'cols':getattr(data,'num_cols',None),
          'self_ref':str(getattr(t,'self_ref',''))})
    return out

def paragraph_items(doc):
    out=defaultdict(list)
    for it in getattr(doc,'texts',[]) or []:
        if str(getattr(it,'label','')).lower().split('.')[-1] not in (
            'text','paragraph','list_item'): continue
        text=str(getattr(it,'text','') or '').strip()
        if len(text)<80: continue
        for prov in (getattr(it,'prov',None) or []):
            if prov.page_no is not None:
                out[int(prov.page_no)].append(text)
                break
    return out

def original_para_match(old_excerpt, new_paragraph):
    if not old_excerpt: return False
    orig=norm(old_excerpt)
    para=norm(new_paragraph)
    if not orig or not para: return False
    return orig in para or (len(orig)>=65 and orig[:65] in para)

def audit_one(q, tables, paras):
    ttype=q.get('type')
    spec=q.get('answer_spec')
    base={'qid':q.get('qid'), 'type':ttype, 'page':q.get('page'),
          'old_table_index':(q.get('meta') or {}).get('table_index'),
          'status':None,'candidate_table_indices':[], 'candidate_table_pages':[],
          'same_page_table_count':0,'matching_paragraph_count':0,
          'original_paragraph_found':None,'note':''}
    if not isinstance(spec,dict) or not spec.get('value') or not isinstance(spec.get('context_keys'),list):
        base.update(status='invalid_schema', note='answer_spec invalid')
        return base
    page=q.get('page')
    try: page=int(page)
    except (TypeError,ValueError):
        base.update(status='invalid_schema',note='page invalid')
        return base
    on_page=[t for t in tables if t['page']==page and t['markdown']]
    base['same_page_table_count']=len(on_page)
    all_md=[t for t in tables if t['markdown']]
    if ttype in ('cell_value','table_about'):
        candidates=[t for t in on_page if em_hit(spec,t['markdown'])]
        anywhere=[t for t in all_md if em_hit(spec,t['markdown'])]
        base['candidate_table_indices']=[t['index'] for t in candidates]
        base['candidate_table_pages']=[t['page'] for t in candidates]
        base['content_hits_elsewhere']=[{'index':t['index'],'page':t['page']} for t in anywhere if t['page']!=page][:8]
        if len(candidates)==1: base['status']='verified_structural'
        elif len(candidates)>1: base['status']='ambiguous_multiple_tables'
        else: base['status']='unmatched_latest_parse'
        return base
    if ttype!='context_dependent':
        base.update(status='invalid_schema', note='unknown question type')
        return base
    if any(em_hit(spec,t['markdown']) for t in on_page):
        base.update(status='invariant_violation_latest',note='table-only PASS in latest Docling')
        return base
    t_candidates=[t for t in on_page if has_token(spec['value'],norm(t['markdown']))]
    paragraph_candidates=[p for p in paras.get(page,[]) if has_keys(spec,p) and not em_hit(spec,p)]
    compatible_tables=[t for t in t_candidates if any(
       em_hit(spec,t['markdown']+'\n'+p) and not em_hit(spec,t['markdown'])
       for p in paragraph_candidates)]
    base['candidate_table_indices']=[t['index'] for t in compatible_tables]
    base['candidate_table_pages']=[t['page'] for t in compatible_tables]
    base['matching_paragraph_count']=len(paragraph_candidates)
    excerpt=(q.get('meta') or {}).get('ctx_para_excerpt','')
    matched_old=[p for p in paragraph_candidates if original_para_match(excerpt,p)]
    base['original_paragraph_found']=bool(matched_old) if excerpt else None
    anywhere=[t for t in all_md if has_token(spec['value'],norm(t['markdown']))]
    base['value_table_hits_elsewhere']=[{'index':t['index'],'page':t['page']}
      for t in anywhere if t['page']!=page][:8]
    if not compatible_tables or not paragraph_candidates:
        base['status']='unmatched_latest_parse'
    elif len(compatible_tables)>1 or len(paragraph_candidates)>1:
        base['status']='ambiguous_evidence'
    elif excerpt and not matched_old:
        base['status']='needs_review_paragraph_changed'
    elif not excerpt:
        base['status']='needs_review_no_original_excerpt'
    else:
        base['status']='verified_structural'
    return base

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--pdf-dir',type=Path,required=True)
    ap.add_argument('--qa-dir',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--max-pdfs',type=int,default=3)
    ap.add_argument('--doc-ids',nargs='*',default=None)
    ap.add_argument('--docling-version',required=True)
    ap.add_argument('--docling-core-version',required=True)
    ap.add_argument('--evidence-sha',required=True)
    args=ap.parse_args()
    assert_isolation()
    assert version('docling')==args.docling_version
    assert version('docling-core')==args.docling_core_version
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    assert_isolation()
    opts=PdfPipelineOptions()
    options_manifest={'pdf_pipeline_options':opts.model_dump(mode='json'),
       'constructor':'DocumentConverter(format_options={"pdf":PdfFormatOption(pipeline_options=PdfPipelineOptions())})',
       'scoring':'generator_3.4.0_normalize_has_token_em_hit',
       'bbox_or_similarity_gate':False,
       'caption_used_as_qa_criterion':False}
    converter=DocumentConverter(format_options={'pdf':PdfFormatOption(pipeline_options=opts)})
    qa_files=sorted((p for p in args.qa_dir.glob('*_qa.json') if not p.name.startswith('_')),
        key=lambda p:(int(re.match(r'^(\d+)',p.name).group(1)) if re.match(r'^(\d+)',p.name) else 999999,p.name))
    if args.doc_ids:
        wanted=set(args.doc_ids)
        qa_files=[p for p in qa_files if p.name[:-len('_qa.json')] in wanted]
        missing=wanted-{p.name[:-len('_qa.json')] for p in qa_files}
        if missing: print('[WARN] requested QA files not found:',sorted(missing),flush=True)
    if args.max_pdfs and args.max_pdfs>0: qa_files=qa_files[:args.max_pdfs]
    args.out.mkdir(parents=True,exist_ok=True)
    details, docs=[],[]
    total_questions=0
    for index,qa_file in enumerate(qa_files,1):
        doc_id=qa_file.name[:-len('_qa.json')]
        pdf=args.pdf_dir/(doc_id+'.pdf')
        print(f'[{index}/{len(qa_files)}] {doc_id}',flush=True)
        if not pdf.exists():
            docs.append({'doc_id':doc_id,'status':'pdf_missing','qa_path':str(qa_file)})
            continue
        try:
            qa_list=json.loads(qa_file.read_text(encoding='utf-8'))
            if not isinstance(qa_list,list): raise ValueError('QA top-level must be list')
        except Exception as e:
            docs.append({'doc_id':doc_id,'status':'qa_load_error','error':str(e)})
            continue
        total_questions+=len(qa_list)
        t0=time.perf_counter()
        try:
            doc=converter.convert(str(pdf)).document
            tables=table_items(doc)
            paras=paragraph_items(doc)
            items=[audit_one(q,tables,paras) for q in qa_list]
            for item in items: item['doc_id']=doc_id
            details.extend(items)
            doc_out={'doc_id':doc_id,'status':'parsed','n_pages':len(doc.pages),
                'n_tables':len(tables),'n_qa':len(qa_list),
                'n_paragraphs':sum(map(len,paras.values())),
                'parse_audit_seconds':round(time.perf_counter()-t0,3),
                'pdf_sha256':sha256(pdf),'qa_sha256':sha256(qa_file),
                'status_counts':dict(Counter(x['status'] for x in items))}
            docs.append(doc_out)
            print('  tables=',len(tables),'QA=',len(items),'statuses=',doc_out['status_counts'],flush=True)
        except Exception as e:
            docs.append({'doc_id':doc_id,'status':'parse_or_audit_error','error':repr(e),
              'traceback':traceback.format_exc()[-4000:]})
            for q in qa_list:
                details.append({'doc_id':doc_id,'qid':q.get('qid'),'type':q.get('type'),
                   'status':'parse_or_audit_error','page':q.get('page')})
            print(' ERROR:',repr(e),flush=True)
        assert_isolation()
        (args.out/'qa_audit_details.json').write_text(json.dumps(details,ensure_ascii=False,indent=2),encoding='utf-8')
        (args.out/'pdf_audit_summary.json').write_text(json.dumps(docs,ensure_ascii=False,indent=2),encoding='utf-8')
    by_status=dict(Counter(r['status'] for r in details))
    by_type={t:dict(Counter(x['status'] for x in details if x.get('type')==t))
        for t in sorted(set(x.get('type') for x in details if x.get('type')))}
    summary={'docling_version':version('docling'),'docling_core_version':version('docling-core'),
      'evidence_chunker_target_sha_NOT_IMPORTED':args.evidence_sha,
      'n_qa_files_selected':len(qa_files),'n_questions_loaded':total_questions,
      'n_questions_audited':len(details),'by_status':by_status,'by_type':by_type,
      'n_pdf_parsed':sum(d['status']=='parsed' for d in docs),
      'n_pdf_failed':sum(d['status']!='parsed' for d in docs),
      'note':'Source-only structural check; not human truth annotation; no deletion or auto QA rewriting; do not postselect on EU or baseline performance'}
    for name,obj in [('qa_audit_summary.json',summary),('parser_config.json',options_manifest)]:
        (args.out/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')
    mappings=[r for r in details if r['status']=='verified_structural']
    with (args.out/'unique_table_match_candidates.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=['doc_id','qid','type','page','old_table_index','new_table_index','status'])
        w.writeheader()
        for r in mappings:
            ids=r.get('candidate_table_indices',[])
            if len(ids)!=1: continue
            w.writerow({'doc_id':r['doc_id'],'qid':r['qid'],'type':r['type'],'page':r['page'],
                'old_table_index':r.get('old_table_index'),'new_table_index':ids[0],
                'status':r['status']})
    print('\nAUDIT FINISHED:',json.dumps(summary,ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':
    main()
