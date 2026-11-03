#!/usr/bin/env python3
'Paired retrieval and token-budget replay; no generative LLM inference.'
from __future__ import annotations
import argparse, copy, gc, hashlib, json, math, os, platform, re, sys, time, traceback
from collections import Counter, defaultdict
import tiktoken
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
import numpy as np

ARMS = ('base', 'eu', 'eu_no_caption', 'eu_no_context', 'eu_no_both',
        'eu_no_abstract', 'eu_no_repeated_caption_prefix')
K_LIST = (1, 5, 10)
PROMPT_TOP_K = (1, 3, 5)
PROMPT_BUDGETS = (512, 1024, 2048, 4096, 8192)
PROMPT_MAX_RANK = 10
PROMPT_SEPARATOR = '\n\n---\n\n'
TOKEN_ENCODING = 'cl100k_base'
MODEL = 'sentence-transformers/all-MiniLM-L6-v2'
PIN_DOCLING = '2.143.1'
PIN_CORE = '2.105.0'
PIN_EU_SHA = '87a43e4f15e50b3ad27e91769ffe9e1864e06791'
DEV_DOCS = [
 '1. Attention is all you need', '6. DPO', '14. CLIP', '21. T5', '26. MMLU',
 '35. APB', '39. risk sharing', '42. rural housing', '45. gao-25-107649',
 '47. gao-26-107681', '50. gao-26-107884', '52. gao-26-108011',
 '55. gao-26-108116', '60. ieee1', '64. ieee5', '66. ieee7',
 '70. ieee11', '72. ieee13', '80. ssrn-1331573', '85. ssrn-2760631',
]

def sha256(path):
 h=hashlib.sha256()
 with open(path, 'rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
 return h.hexdigest()

def dump_json(path, data):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 tmp=path.with_suffix(path.suffix+'.tmp')
 tmp.write_text(json.dumps(data,ensure_ascii=False,indent=2,default=str),encoding='utf-8')
 tmp.replace(path)

def normalize(s):
 if not s:return ''
 s=str(s).lower().replace('\xa0',' ')
 s=re.sub(r'[^\w.\-±× ]+',' ',s)
 s=re.sub(r'\s+',' ',s).strip()
 return re.sub(r'\.$','',s)

def has_token(needle, normalized_haystack):
 n=normalize(needle)
 return bool(n and re.search(r'(?<!\d)'+re.escape(n)+r'(?!\d)',normalized_haystack))

def evidence_hit(spec,chunk):
 if not isinstance(spec,dict) or not chunk:return False
 text=normalize(chunk)
 return has_token(spec.get('value',''),text) and all(has_token(k,text) for k in spec.get('context_keys',[]))


def prompt_metrics(ordered_entries, spec, tokenizer, budgets=PROMPT_BUDGETS):
    'Measure full returned-context tokens and prefix-fit evidence under fixed budgets.'
    docs = list(ordered_entries[:PROMPT_MAX_RANK])
    chunks = [e['display'] for e in docs]
    prefix_token_costs=[]
    prefix_prompt_hits=[]
    prefix_single_hits=[]
    current = ''
    any_single_hit = False
    for i, chunk in enumerate(chunks):
        current = (current + PROMPT_SEPARATOR + chunk) if i else chunk
        cost = len(tokenizer.encode(current, disallowed_special=()))
        prefix_token_costs.append(cost)
        any_single_hit = any_single_hit or evidence_hit(spec, chunk)
        prefix_single_hits.append(int(any_single_hit))
        prefix_prompt_hits.append(int(evidence_hit(spec, current)))
    if any(a > b for a,b in zip(prefix_token_costs,prefix_token_costs[1:])):
        raise AssertionError('Adding retrieved text decreased exact prompt token cost')
    def get_at_k(seq,k,default=0):
        return seq[min(k,len(seq))-1] if seq else default
    topk_tokens={str(k):get_at_k(prefix_token_costs,k) for k in PROMPT_TOP_K}
    topk_prompt_hit={str(k):get_at_k(prefix_prompt_hits,k) for k in PROMPT_TOP_K}
    topk_single_hit={str(k):get_at_k(prefix_single_hits,k) for k in PROMPT_TOP_K}
    budget_prompt_hit={};budget_single_hit={};budget_used_tokens={};budget_n_chunks={}
    for b in budgets:
        n_fit=sum(int(v<=b) for v in prefix_token_costs)
        if n_fit and prefix_token_costs[n_fit-1]>b:
            raise AssertionError('Budget fit is not a rank prefix')
        budget_prompt_hit[str(b)]=int(prefix_prompt_hits[n_fit-1]) if n_fit else 0
        budget_single_hit[str(b)]=int(prefix_single_hits[n_fit-1]) if n_fit else 0
        budget_used_tokens[str(b)]=int(prefix_token_costs[n_fit-1]) if n_fit else 0
        budget_n_chunks[str(b)]=n_fit
    first_rank=next((i+1 for i,h in enumerate(prefix_prompt_hits) if h),None)
    first_single_rank=next((i+1 for i,h in enumerate(prefix_single_hits) if h),None)
    return {
        'topk_context_tokens':topk_tokens,
        'topk_prompt_evidence_hit':topk_prompt_hit,
        'topk_single_chunk_evidence_hit':topk_single_hit,
        'budget_prompt_evidence_hit':budget_prompt_hit,
        'budget_single_chunk_evidence_hit':budget_single_hit,
        'budget_used_context_tokens':budget_used_tokens,
        'budget_n_chunks':budget_n_chunks,
        'first_prompt_evidence_rank':first_rank,
        'first_prompt_evidence_tokens':prefix_token_costs[first_rank-1] if first_rank else None,
        'first_single_chunk_evidence_rank':first_single_rank,
        'first_single_chunk_evidence_tokens':prefix_token_costs[first_single_rank-1] if first_single_rank else None,
        'top1_exceeds_512_context_tokens':int(bool(prefix_token_costs) and prefix_token_costs[0]>512),
    }


def paired_token_cost_stats(rows, arm_a, arm_b, k, rounds=10000, seed=20261029):
    """Paired PDF-cluster bootstrap for mean LLM context token cost."""
    matched=[r for r in rows if arm_a in r['arms'] and arm_b in r['arms']]
    if not matched:return None
    key=str(k)
    av=np.array([r['arms'][arm_a]['prompt']['topk_context_tokens'][key] for r in matched],dtype=float)
    bv=np.array([r['arms'][arm_b]['prompt']['topk_context_tokens'][key] for r in matched],dtype=float)
    doc=defaultdict(lambda:[0,0.0])
    for r, a, b in zip(matched,av,bv):
        it=doc[r['doc_id']];it[0]+=1;it[1]+=float(b-a)
    sample=np.array(list(doc.values()),dtype=float)
    ci=None
    if len(sample)>1:
        rng=np.random.default_rng(seed);out=[]
        for off in range(0,rounds,512):
            inds=rng.integers(0,len(sample),size=(min(512,rounds-off),len(sample)))
            sums=sample[inds].sum(axis=1)
            out.extend((sums[:,1]/sums[:,0]).tolist())
        ci=[round(float(v),3) for v in np.percentile(out,[2.5,97.5])]
    return {'k':k,'n':len(matched),'arm_a':arm_a,'arm_b':arm_b,
        'mean_a_tokens':round(float(av.mean()),3),
        'mean_b_tokens':round(float(bv.mean()),3),
        'mean_delta_b_minus_a_tokens':round(float((bv-av).mean()),3),
        'median_a_tokens':round(float(np.median(av)),3),
        'median_b_tokens':round(float(np.median(bv)),3),
        'document_cluster_bootstrap_ci95_delta_tokens':ci,
        'rounds':rounds,'seed':seed}


def summarize_prompt_metrics(rows,arms=ARMS):
    """Report coverage and costs together; don't compare only successful cases."""
    output={'measurement':{
      'encoding':TOKEN_ENCODING,
      'counted_content':'Retrieved display chunk texts and fixed \n\n---\n\n separators only',
      'excluded':'System instructions, user question, chat template, citation wrapper, generated answer, actual API billing',
      'max_rank':PROMPT_MAX_RANK,
      'budget_policy':'Take ranked unique chunks in order; STOP before first chunk that exceeds the total context-only budget. No splitting, skipping, summarization, or reranking.',
      'full_chunk_constraint':'An oversized top1 chunk yields zero selected chunks for that budget.',
      'prompt_hit':'Existing value plus all context_keys appear anywhere in CONCATENATED retrieved context; they may occur in different chunks (proxy, not answer correctness).',
      'single_chunk_hit':'At least one individually returned display chunk passes the unmodified EvidenceHit value+context_keys rule.',
      'token_saving_claim':'Not established unless compared at matched success/coverage and budget.',
    },'by_arm':{},'paired_budget_prompt_hit':{},'paired_topk_prompt_hit':{},
    'paired_topk_token_cost':{},'by_type':{}}
    for arm in arms:
        prompts=[r['arms'][arm]['prompt'] for r in rows]
        topk={};budget={}
        for k in PROMPT_TOP_K:
            key=str(k)
            costs=np.array([p['topk_context_tokens'][key] for p in prompts],dtype=float)
            topk[key]={
              'mean_context_tokens':round(float(costs.mean()),3),
              'median_context_tokens':round(float(np.median(costs)),3),
              'p90_context_tokens':round(float(np.percentile(costs,90)),3),
              'prompt_evidence_hit_rate':round(float(np.mean([p['topk_prompt_evidence_hit'][key] for p in prompts])),6),
              'single_chunk_evidence_hit_rate':round(float(np.mean([p['topk_single_chunk_evidence_hit'][key] for p in prompts])),6),
            }
        for b in PROMPT_BUDGETS:
            key=str(b)
            budget[key]={
                'prompt_evidence_hit_rate':round(float(np.mean([p['budget_prompt_evidence_hit'][key] for p in prompts])),6),
                'single_chunk_evidence_hit_rate':round(float(np.mean([p['budget_single_chunk_evidence_hit'][key] for p in prompts])),6),
                'mean_context_tokens_used':round(float(np.mean([p['budget_used_context_tokens'][key] for p in prompts])),3),
                'mean_chunks_fit':round(float(np.mean([p['budget_n_chunks'][key] for p in prompts])),3),
                'n_zero_chunks_fit':sum(p['budget_n_chunks'][key]==0 for p in prompts),
            }
        found=[p['first_prompt_evidence_tokens'] for p in prompts if p['first_prompt_evidence_tokens'] is not None]
        output['by_arm'][arm]={
           'topk':topk,'budget':budget,
           'first_evidence_within_top10':{
             'n_found':len(found), 'coverage':round(len(found)/len(prompts),6),
             'median_tokens_among_found_only':round(float(np.median(found)),3) if found else None,
             'p90_tokens_among_found_only':round(float(np.percentile(found,90)),3) if found else None,
             'warning':'Conditional cost among successes: never compare alone without coverage.'
           },
           'top1_exceeds_512_rate':round(float(np.mean([p['top1_exceeds_512_context_tokens'] for p in prompts])),6),
        }
    comparisons=(('base','eu'),('eu','eu_no_repeated_caption_prefix'),('base','eu_no_repeated_caption_prefix'))
    for a,b in comparisons:
        name=f'{a}__vs__{b}'
        prompt_rows=[]
        for r in rows:
            paired={**r,'arms':{
              arm:{**vals,
                 'budget_prompt_evidence_hit':vals['prompt']['budget_prompt_evidence_hit'],
                 'topk_prompt_evidence_hit':vals['prompt']['topk_prompt_evidence_hit']}
              for arm,vals in r['arms'].items()}}
            prompt_rows.append(paired)
        output['paired_budget_prompt_hit'][name]={str(budget):paired_stats(prompt_rows,a,b,'budget_prompt_evidence_hit',budget)
          for budget in PROMPT_BUDGETS}
        output['paired_topk_prompt_hit'][name]={str(k):paired_stats(prompt_rows,a,b,'topk_prompt_evidence_hit',k)
          for k in PROMPT_TOP_K}
        output['paired_topk_token_cost'][name]={str(k):paired_token_cost_stats(rows,a,b,k)
          for k in PROMPT_TOP_K}
    for typ in sorted({r['type'] for r in rows}):
        rt=[r for r in rows if r['type']==typ]
        output['by_type'][typ]={'n':len(rt),'budget_prompt_evidence_hit':{
          arm:{str(b):round(float(np.mean([r['arms'][arm]['prompt']['budget_prompt_evidence_hit'][str(b)] for r in rt])),6)
               for b in PROMPT_BUDGETS} for arm in ('base','eu','eu_no_repeated_caption_prefix')},
          'topk_prompt_evidence_hit':{
          arm:{str(k):round(float(np.mean([r['arms'][arm]['prompt']['topk_prompt_evidence_hit'][str(k)] for r in rt])),6)
               for k in PROMPT_TOP_K} for arm in ('base','eu','eu_no_repeated_caption_prefix')}}
    return output

def paired_stats(rows, arm_a, arm_b, metric, k=None, rounds=10000, seed=20261029):
 matched=[r for r in rows if arm_a in r['arms'] and arm_b in r['arms']]
 if not matched:return None
 def val(r,a):
  d=r['arms'][a]
  return int(d[metric] if k is None else d[metric][str(k)])
 av=np.array([val(r,arm_a) for r in matched], dtype=np.int8)
 bv=np.array([val(r,arm_b) for r in matched], dtype=np.int8)
 docs=defaultdict(lambda: [0,0]);
 for r,x,y in zip(matched,av,bv):
  z=docs[r['doc_id']];z[0]+=1;z[1]+=int(y)-int(x)
 sample=np.array(list(docs.values()),dtype=np.int64)
 ci=None
 if len(sample)>1:
  rng=np.random.default_rng(seed);results=[]
  for off in range(0,rounds,512):
   ix=rng.integers(0,len(sample),size=(min(512,rounds-off),len(sample)))
   s=sample[ix].sum(axis=1)
   results.extend((100*s[:,1]/s[:,0]).tolist())
  ci=[round(x,4) for x in np.percentile(results,[2.5,97.5])]
 return {'n':len(matched),'n_documents':len(docs),'arm_a':arm_a,'arm_b':arm_b,
  'metric':metric+('('+str(k)+')' if k is not None else ''),
  'arm_a_rate':round(float(av.mean()),6),'arm_b_rate':round(float(bv.mean()),6),
  'delta_b_minus_a_pp':round(float(100*(bv.astype(float)-av).mean()),4),
  'document_cluster_bootstrap_ci95_pp':ci,'bootstrap_rounds':rounds,'seed':seed,
  'both_hit':int(((av==1)&(bv==1)).sum()),
  'a_only':int(((av==1)&(bv==0)).sum()),
  'b_only':int(((av==0)&(bv==1)).sum()),
  'neither':int(((av==0)&(bv==0)).sum())}

def chunk_pages(c):
 pages=set()
 for di in (getattr(getattr(c,'meta',None),'doc_items',[]) or []):
  for prov in (getattr(di,'prov',[]) or []):
   if getattr(prov,'page_no',None) is not None:pages.add(int(prov.page_no))
 return pages

def baseline_table_indices(c, tables):
 by_ref={str(t.self_ref):i for i,t in enumerate(tables)}
 out=set()
 for it in (getattr(getattr(c,'meta',None),'doc_items',[]) or []):
  ref=str(getattr(it,'self_ref',''))
  if ref in by_ref:out.add(by_ref[ref])
  else:
   for i,t in enumerate(tables):
    if it is t:out.add(i)
 return out

def get_eu_table_index(c,doc_tables,parse_warnings):
 meta=c.metadata or {}
 idx=meta.get('table_index')
 if idx is None:return set()
 try:idx=int(idx)
 except (TypeError,ValueError):
  parse_warnings.append({'kind':'invalid_table_index','value':str(idx)});return set()
 if not (0<=idx<len(doc_tables)):
  parse_warnings.append({'kind':'out_of_range_eu_table_index','index':idx});return set()
 pages=meta.get('page_span') or [meta.get('page_no')]
 if isinstance(pages,(int,str)):pages=[pages]
 raw_pg=(doc_tables[idx].prov[0].page_no if doc_tables[idx].prov else None)
 try: page_set={int(p) for p in pages if p is not None}
 except (TypeError,ValueError):page_set=set()
 if raw_pg is not None and page_set and int(raw_pg) not in page_set:
  parse_warnings.append({'kind':'eu_index_page_mismatch','index':idx,
                         'eu_pages':sorted(page_set),'doc_table_page':int(raw_pg)})
  return set()
 return {idx}

def ablate(eus, caption=False, context=False, abstract=False):
    'Change only the selected EvidenceUnit fields for each ablation.'
    if abstract and (caption or context):
        raise ValueError('abstract ablation is a separate, independent arm')
    result=[]
    for eu in eus:
        e=copy.deepcopy(eu)
        if caption:
            e.table_abstract=eu.safe_abstract
            e.caption_text=None
            e.caption_confidence='none'
        if context:
            e.context_before=[]
            e.context_after=[]
            e.page_span={e.page_no}
        if abstract:
            e.table_abstract=None
        result.append(e)
    return result


def check_ablation_invariants(original, variants):
    """Hard checks against accidental co-removal of caption/context/abstract."""
    common=('table_html','flattened_rows','footnote_text','section_header',
            'table_index','page_no','eu_id','row_sentence_map')
    n=0
    for arm,targets in variants.items():
        if len(targets)!=len(original):
            raise AssertionError(f'{arm}: EvidenceUnit cardinality changed')
        cap=arm in ('eu_no_caption','eu_no_both')
        ctx=arm in ('eu_no_context','eu_no_both')
        abs_only=arm=='eu_no_abstract'
        for src,e in zip(original,targets):
            for field in common:
                if getattr(e,field)!=getattr(src,field):
                    raise AssertionError(f'{arm} modified unrelated {field}: {src.eu_id}')
            if cap:
                if e.caption_text is not None or e.safe_abstract!=src.safe_abstract:
                    raise AssertionError(f'{arm} removed derived header/row summary: {src.eu_id}')
            elif e.caption_text!=src.caption_text:
                raise AssertionError(f'{arm} unexpectedly changed caption: {src.eu_id}')
            if ctx:
                if e.context_before or e.context_after or e.page_span!={e.page_no}:
                    raise AssertionError(f'{arm} did not fully remove context fields: {src.eu_id}')
            elif (e.context_before!=src.context_before or
                  e.context_after!=src.context_after or e.page_span!=src.page_span):
                raise AssertionError(f'{arm} unexpectedly changed context: {src.eu_id}')
            if abs_only:
                if e.table_abstract is not None:
                    raise AssertionError(f'{arm} did not remove abstract: {src.eu_id}')
            elif not cap and e.table_abstract!=src.table_abstract:
                raise AssertionError(f'{arm} unexpectedly changed abstract: {src.eu_id}')
            n+=1
    return n


def ablation_cross_field_overlaps(original, variants):
    'Report semantic overlap, not a hard failure/automatic leakage claim.'
    overlaps=[]
    for arm,eus in variants.items():
        drop_caption=arm in ('eu_no_caption','eu_no_both')
        drop_context=arm in ('eu_no_context','eu_no_both')
        if not (drop_caption or drop_context):
            continue
        for src,dst in zip(original,eus):
            removed=[]
            if drop_caption and src.caption_text:
                removed.append(('caption',str(src.caption_text)))
            if drop_context:
                removed += [('context',str(v)) for v in (src.context_before or [])+(src.context_after or [])]
            retained={
                'table_html':str(dst.table_html or ''),
                'flattened_rows':'\n'.join(map(str,dst.flattened_rows)),
                'footnote_text':str(dst.footnote_text or ''),
                'section_header':str(dst.section_header or ''),
                'table_abstract':str(dst.table_abstract or ''),
                'context_before':'\n'.join(map(str,dst.context_before)),
                'context_after':'\n'.join(map(str,dst.context_after)),
                'caption_text':str(dst.caption_text or ''),
            }
            for kind,value in removed:
                needle=normalize(value)
                if len(needle)<50:continue
                for channel,raw in retained.items():
                    if needle in normalize(raw):
                        overlaps.append({'arm':arm,'removed_field':kind,
                            'remaining_field':channel,'eu_id':str(src.eu_id),
                            'length':len(needle),'excerpt':value[:100],
                            'interpretation':'semantic_overlap_in_independent_field'})
    return overlaps


def remove_repeated_caption_prefix(c, unit):
    'Remove caption prefixes from embedding input, leaving displayed evidence unchanged.'
    caption=getattr(c,'safe_caption',None)
    if not caption:return str(unit)
    prefix=f'[{caption}] '
    u=str(unit)
    return u[len(prefix):] if u.startswith(prefix) and u[len(prefix):] else u

def to_entries(doc, base_chunks, eu_chunks, warning_list):
 doc_tables=doc.tables
 base=[]
 for idx,c in enumerate(base_chunks):
  base.append({'id':f'base:{idx}','retrieval':str(c.text), 'display':str(c.text),
   'pages':chunk_pages(c),'tables':baseline_table_indices(c,doc_tables)})
 results={'base':base}
 for name,chunks in eu_chunks.items():
  entries=[]
  for j,c in enumerate(chunks):
   md=c.metadata or {};pg=md.get('page_no');p=md.get('page_span') or ([pg] if pg is not None else [])
   if isinstance(p,(int,str)):p=[p]
   ps=set()
   for pi in p:
    try:ps.add(int(pi))
    except (TypeError,ValueError):pass
   ti=get_eu_table_index(c,doc_tables,warning_list) if md.get('table_index') is not None else set()
   rid=str(c.chunk_id)
   for u in (c.retrieval_units or []):
    if u:
     search_text=(remove_repeated_caption_prefix(c,u)
                  if name=='eu_no_repeated_caption_prefix' else str(u))
     entries.append({'id':rid,'retrieval':search_text,'display':str(c.text),
                     'pages':ps,'tables':ti})
  results[name]=entries
 if 'eu_no_repeated_caption_prefix' in results and 'eu' in results:
  a=results['eu'];b=results['eu_no_repeated_caption_prefix']
  if len(a)!=len(b):raise AssertionError('Prefix-off changed retrieval vector count')
  for x,y in zip(a,b):
   for key in ('id','display','pages','tables'):
    if x[key]!=y[key]:raise AssertionError(f'Prefix-off changed {key}')
 return results

def token_length_diagnostic(model,entries,max_samples=256):
    'Deterministic sample of raw embedding input lengths (before truncation).'
    if not entries:return {'n_sampled':0}
    tokenizer=model.tokenizer
    indices=np.linspace(0,len(entries)-1,num=min(max_samples,len(entries)),dtype=int)
    strings=[entries[int(i)]['retrieval'] for i in indices]
    token_ids=tokenizer(strings,add_special_tokens=True,truncation=False,padding=False)['input_ids']
    lengths=np.array([len(v) for v in token_ids],dtype=np.int32)
    limit=int(model.max_seq_length)
    return {'n_sampled':len(lengths),'total_candidates':len(entries),
            'sampling':'evenly_spaced_by_vector_index',
            'model_max_seq_length':limit,
            'n_over_model_limit':int(np.sum(lengths>limit)),
            'fraction_over_limit':round(float(np.mean(lengths>limit)),4),
            'median_tokens':float(np.median(lengths)),
            'p95_tokens':float(np.percentile(lengths,95)),
            'max_tokens_sampled':int(max(lengths))}

def rank(scores, entries, max_k=10, unique_chunk=False):
 if not len(scores):return []
 if not unique_chunk:
  return np.argsort(-np.asarray(scores),kind='stable')[:max_k].tolist()
 best={}
 for i,(score,e) in enumerate(zip(scores,entries)):
  cid=e['id'];sc=float(score)
  if cid not in best or sc>best[cid][0]:best[cid]=(sc,i)
 return [i for _,i in sorted(best.values(),key=lambda z:(-z[0],z[1]))[:max_k]]

def score_rows(qa, model, entries, device, batch_size, prompt_tokenizer):
 import torch
 def sync():
  if device=='cuda':torch.cuda.synchronize()
 model_embedding_time={};embeddings={}
 for arm,vec in entries.items():
  if not vec:raise RuntimeError(f'Empty corpus in {arm}; cannot produce a paired result')
  texts=[x['retrieval'] for x in vec]
  sync();t=time.perf_counter()
  emb=model.encode(texts,normalize_embeddings=True,batch_size=batch_size,show_progress_bar=False)
  sync();model_embedding_time[arm]=time.perf_counter()-t
  embeddings[arm]=np.asarray(emb,dtype=np.float32)
 sync();t=time.perf_counter()
 questions=model.encode([x['question_delabeled'] for x in qa],normalize_embeddings=True,
    batch_size=batch_size,show_progress_bar=False)
 sync();query_embed=time.perf_counter()-t
 q=np.asarray(questions,dtype=np.float32)
 scores_all={};ranking_time={}
 for arm,mat in embeddings.items():
  sync();t=time.perf_counter()
  score=q@mat.T
  orders=[rank(row,entries[arm],10,arm!='base') for row in score]
  sync();ranking_time[arm]=time.perf_counter()-t
  scores_all[arm]=orders
 rows=[]
 for i,orig in enumerate(qa):
  gold_idx=(orig.get('meta') or {}).get('table_index')
  if gold_idx is not None:gold_idx=int(gold_idx)
  gold_pg=int(orig['page'])
  row={'doc_id':orig['doc_id'],'qid':orig['qid'],'type':orig['type'],
       'page':gold_pg,'gold_table_index':gold_idx,
       'dist_pt':(orig.get('meta') or {}).get('dist_pt'),
       'arms':{}}
  for arm,order_list in scores_all.items():
   order=order_list[i];docs=[entries[arm][j] for j in order]
   hits={str(k):int(any(gold_pg in d['pages'] for d in docs[:k])) for k in K_LIST}
   tblhits={str(k):int(any(gold_idx in d['tables'] for d in docs[:k])) for k in K_LIST} if gold_idx is not None else None
   top=docs[0]['display'] if docs else ''
   row['arms'][arm]={'page_hit':hits,'table_hit':tblhits,
       'evidence_hit_1':int(evidence_hit(orig['answer_spec'],top)),
       'top1_page':sorted(docs[0]['pages']) if docs else [],
       'top1_tables':sorted(docs[0]['tables']) if docs else [],
       'prompt':prompt_metrics(docs,orig['answer_spec'],prompt_tokenizer)}
  rows.append(row)
 return rows,model_embedding_time,query_embed,ranking_time

def _sort_key(p):
 m=re.match(r'^(\d+)\.',p.name)
 return (int(m.group(1)) if m else 999999,p.name)

def preflight(pdf_dir,qa_dir,audit_pdf_summary,expected_n=None):
 docs=json.loads(Path(audit_pdf_summary).read_text(encoding='utf-8'))
 by_name={d['doc_id']:d for d in docs}
 if len(by_name)!=90:raise RuntimeError('Audit manifest must contain 90 documents')
 matched=[]
 for doc_id,d in by_name.items():
  pdf=pdf_dir/(doc_id+'.pdf');qa=qa_dir/(doc_id+'_qa.json')
  if not pdf.is_file() or not qa.is_file():raise FileNotFoundError(str(pdf)+' or '+str(qa))
  if sha256(pdf)!=d['pdf_sha256']:raise RuntimeError('PDF SHA changed since audit: '+doc_id)
  matched.append((pdf,qa,doc_id,d))
 assert len(matched)==90
 count=sum(len(json.loads(q.read_text(encoding='utf-8'))) for _,q,_,_ in matched)
 if expected_n is not None and count!=expected_n:raise RuntimeError(f'QA count {count}, expected {expected_n}')
 return sorted(matched,key=lambda x:_sort_key(x[0]))

def main():
 ap=argparse.ArgumentParser()
 ap.add_argument('--pdf-dir',type=Path,required=True)
 ap.add_argument('--qa-dir',type=Path,required=True)
 ap.add_argument('--audit-dir',type=Path,required=True)
 ap.add_argument('--out-dir',type=Path,required=True)
 ap.add_argument('--repo-dir',type=Path,required=True)
 ap.add_argument('--model-sha',required=True)
 ap.add_argument('--max-pdfs',type=int,default=3)
 ap.add_argument('--dev-only',action='store_true')
 ap.add_argument('--bbox',type=float,default=300)
 ap.add_argument('--sim',type=float,default=0)
 ap.add_argument('--batch-size',type=int,default=64)
 args=ap.parse_args()
 if version('docling')!=PIN_DOCLING or version('docling-core')!=PIN_CORE:
  raise RuntimeError('Docling versions differ from QA audit!')
 import subprocess
 got=subprocess.check_output(['git','-C',str(args.repo_dir),'rev-parse','HEAD'],text=True).strip()
 if got!=PIN_EU_SHA:raise RuntimeError('Evidence-Chunker SHA mismatch: '+got)
 import importlib
 pinned_src=(args.repo_dir/'src').resolve()
 expected_init=(pinned_src/'evidence_chunker'/'__init__.py')
 if not expected_init.is_file():raise RuntimeError('Pinned source package missing: '+str(expected_init))
 if str(pinned_src) not in sys.path:sys.path.insert(0,str(pinned_src))
 importlib.invalidate_caches()
 import evidence_chunker
 if Path(evidence_chunker.__file__).resolve()!=expected_init:
  raise RuntimeError('Imported wrong evidence_chunker source: '+str(evidence_chunker.__file__))
 from evidence_chunker.parser.docling import DoclingParser
 from evidence_chunker.chunker import build_evidence_units
 from evidence_chunker.split import split_oversized_units
 from evidence_chunker.export import TextChunk, filter_consumed_paragraphs
 from docling.document_converter import DocumentConverter, PdfFormatOption
 from docling.datamodel.pipeline_options import PdfPipelineOptions
 from docling.chunking import HybridChunker
 from docling_core.types.doc import DocItemLabel
 from sentence_transformers import SentenceTransformer
 import torch
 opts=PdfPipelineOptions()
 expected=json.loads((args.audit_dir/'parser_config.json').read_text())['pdf_pipeline_options']
 actual=opts.model_dump(mode='json')
 def canonical_parser_options(d):
  normalized=copy.deepcopy(d)
  chart=normalized.get('chart_extraction_options') or {}
  spec=chart.get('model_spec') or {}
  engines=spec.get('supported_engines')
  if isinstance(engines,list) and all(isinstance(v,str) for v in engines):
   spec['supported_engines']=sorted(engines)
  return normalized
 def option_differences(x,y,path=''):
  if type(x) is not type(y):return [(path or '<root>',x,y)]
  if isinstance(x,dict):
   out=[]
   for key in sorted(set(x)|set(y)):
    p=f'{path}.{key}' if path else key
    if key not in x or key not in y:out.append((p,x.get(key,'<MISSING>'),y.get(key,'<MISSING>')))
    else:out.extend(option_differences(x[key],y[key],p))
   return out
  if isinstance(x,list):
   if len(x)!=len(y):return [(path,x,y)]
   out=[]
   for i,(a,b) in enumerate(zip(x,y)):
    out.extend(option_differences(a,b,f'{path}[{i}]'))
   return out
  return [] if x==y else [(path,x,y)]
 exp_canon=canonical_parser_options(expected)
 act_canon=canonical_parser_options(actual)
 changes=option_differences(exp_canon,act_canon)
 if changes:
  print('[Parser config mismatch] Exact differences after canonicalizing nonoperative supported_engines:',flush=True)
  for field,old,new in changes[:20]:
   print(f'  {field}: audit={str(old)[:160]} | current={str(new)[:160]}',flush=True)
  raise RuntimeError(f'Parser options changed since QA audit: {len(changes)} field differences')
 if expected!=actual:
  print('[Parser config] PASS: only chart model supported_engines ordering differs; no functional option drift.',flush=True)
 else:
  print('[Parser config] PASS: complete serialized options match QA audit.',flush=True)
 converter=DocumentConverter(format_options={'pdf':PdfFormatOption(pipeline_options=opts)})
 device='cuda' if torch.cuda.is_available() else 'cpu'
 print('Loading embedding model',MODEL,args.model_sha,'device=',device,flush=True)
 model=SentenceTransformer(MODEL,revision=args.model_sha,device=device)
 model.max_seq_length=min(int(model.max_seq_length),512)
 prompt_tokenizer=tiktoken.get_encoding(TOKEN_ENCODING)
 print('Prompt tokenizer:',TOKEN_ENCODING,'max rank',PROMPT_MAX_RANK,
       'budgets',PROMPT_BUDGETS,flush=True)
 all_pairs=preflight(args.pdf_dir,args.qa_dir,args.audit_dir/'pdf_audit_summary.json',2239)
 if args.dev_only:all_pairs=[p for p in all_pairs if p[2] in DEV_DOCS]
 if args.max_pdfs and args.max_pdfs>0:all_pairs=all_pairs[:args.max_pdfs]
 if not all_pairs:raise RuntimeError('No PDF and QA pairs')
 args.out_dir.mkdir(parents=True,exist_ok=True)
 run_info={'docling':version('docling'),'docling_core':version('docling-core'),
  'evidence_chunker':version('evidence-chunker'),'evidence_git_sha':got,
  'embed_model':MODEL,'embed_model_sha':args.model_sha,
  'parser_options':actual,'bbox':args.bbox,'sim':args.sim,
  'batch_size':args.batch_size,'embedding_max_seq_length':model.max_seq_length,'device':device,'gpu':torch.cuda.get_device_name(0) if device=='cuda' else None,
  'python':platform.python_version(),'selected_docs':len(all_pairs),
  'scope':'dev20' if args.dev_only else 'full90','run_started_utc':datetime.now(timezone.utc).isoformat(),
  'metric_note':'PageHit=source page provenance; TableHit=source table index; EvidenceHit=legacy value+context_keys in one display chunk; prompt proxy measures concatenated evidence only; no generated LLM',
  'prompt_token_metrics':{'encoding':TOKEN_ENCODING,
    'budgets':PROMPT_BUDGETS,'top_k':PROMPT_TOP_K,
    'max_rank':PROMPT_MAX_RANK,
    'policy':'Full chunks in rank order; stop before first oversized chunk; no skipping/splitting; exact cl100k_base encoding of concatenated display text with fixed separator'},
  'ablation_note':'Field-level factorial ablations: caption removes caption_text + its literal contribution from derived abstract (other abstract facts retained); context removes before/after only; abstract-only separately. Prefix-off changes only embedding retrieval prefixes. Cross-field semantic overlaps separately reported, not silently erased.',
  'latency_note':'Docling parsed once; hybrid and EU chunk sequential; embedding/ranking sequential, no strict isolated CPU/GPU benchmark.'}
 dump_json(args.out_dir/'run_config.json',run_info)
 full_rows=[];profiles=[];failures=[];warnings=[];leakage=[]
 for j,(pdf,qa_path,doc_id,auditdoc) in enumerate(all_pairs,1):
  print(f'\n[{j}/{len(all_pairs)}] {doc_id}',flush=True)
  timings={}
  try:
   st=time.perf_counter();doc=converter.convert(str(pdf)).document
   timings['parse_s']=time.perf_counter()-st
   if len(doc.pages)!=auditdoc['n_pages'] or len(doc.tables)!=auditdoc['n_tables']:
    raise RuntimeError(f'Parser drift: audit pages/tables={auditdoc["n_pages"]}/{auditdoc["n_tables"]}; now={len(doc.pages)}/{len(doc.tables)}')
   qlist=json.loads(qa_path.read_text(encoding='utf-8'))
   if not qlist:raise RuntimeError('QA file contains no verified questions')
   for a in qlist:
    a['doc_id']=doc_id
    ix=int((a.get('meta') or {})['table_index']);pg=int(a['page'])
    if not (0<=ix<len(doc.tables)) or not doc.tables[ix].prov or doc.tables[ix].prov[0].page_no!=pg:
     raise RuntimeError('QA↔Docling table mapping drift: '+str(a['qid']))
   st=time.perf_counter();hc=list(HybridChunker().chunk(doc))
   timings['hybrid_chunk_s']=time.perf_counter()-st
   st=time.perf_counter();parsed=DoclingParser().from_doc(doc)
   eus=split_oversized_units(build_evidence_units(parsed,args.bbox,args.sim,doc_id))
   non_table=[c for c in hc if not any(di.label==DocItemLabel.TABLE for di in c.meta.doc_items)]
   non_table=filter_consumed_paragraphs(non_table,eus)
   tc=[]
   for i,c in enumerate(non_table):
    text=TextChunk(c,doc_id,i)
    text.metadata['page_span']=chunk_pages(c)
    tc.append(text)
   timings['eu_chunk_s']=time.perf_counter()-st
   st=time.perf_counter()
   variants={'eu_no_caption':ablate(eus,caption=True),
             'eu_no_context':ablate(eus,context=True),
             'eu_no_both':ablate(eus,caption=True,context=True),
             'eu_no_abstract':ablate(eus,abstract=True)}
   ablation_checks=check_ablation_invariants(eus,variants)
   ablation_issues=ablation_cross_field_overlaps(eus,variants)
   timings['ablation_build_s']=time.perf_counter()-st
   entry_warnings=[]
   table_hc = [c for c in hc if any(getattr(di, 'label', None) == DocItemLabel.TABLE for di in (c.meta.doc_items or []))]
   mapped_hc = [c for c in table_hc if baseline_table_indices(c, doc.tables)]
   if doc.tables and table_hc and not mapped_hc:
    raise RuntimeError('Baseline TableHit metadata unresolved: zero source self_ref matches among table chunks')
   corpora={n:(units+tc) for n,units in variants.items()}
   corpora['eu']=eus+tc
   corpora['eu_no_repeated_caption_prefix']=eus+tc
   entries=to_entries(doc,hc,corpora,entry_warnings)
   table_issue=[w for w in entry_warnings if w.get('kind') in ('eu_index_page_mismatch','out_of_range_eu_table_index','invalid_table_index')]
   if table_issue:
    raise RuntimeError('EU table index provenance not reliably mappable; example '+str(table_issue[0]))
   length_diag={}
   for diag_arm in ('eu','eu_no_caption','eu_no_repeated_caption_prefix'):
    try:length_diag[diag_arm]=token_length_diagnostic(model,entries[diag_arm])
    except Exception as e:length_diag[diag_arm]={'diagnostic_error':str(e)}
   rows,emb_times,query_emb,ranking_times=score_rows(qlist,model,entries,device,args.batch_size,prompt_tokenizer)
   for scored in rows:
    for checked_arm in ARMS:
     if scored['arms'][checked_arm]['evidence_hit_1'] != scored['arms'][checked_arm]['prompt']['topk_prompt_evidence_hit']['1']:
      raise AssertionError('Rank-1 prompt metric drift from frozen EvidenceHit@1')
   timings['embedding_per_arm_s']=emb_times
   timings['question_embedding_s']=query_emb
   timings['ranking_per_arm_s']=ranking_times
   profile={'doc_id':doc_id,'n_qa':len(rows),'n_pages':len(doc.pages),'n_tables':len(doc.tables),
     'n_baseline_chunks':len(hc),'n_eu_chunks':len(eus),
     'n_split_eu':sum(bool(getattr(e,'is_split',False)) for e in eus),
     'max_table_rows':max((getattr(t.data,'num_rows',0) or 0 for t in doc.tables),default=0),
     'max_table_cols':max((getattr(t.data,'num_cols',0) or 0 for t in doc.tables),default=0),
     'corpus_vectors':{name:len(e) for name,e in entries.items()},
     'corpus_unique_chunks':{name:len(set(x['id'] for x in e)) for name,e in entries.items()},
     'timings':timings,'ablation_cross_field_overlap_flags':len(ablation_issues),
     'ablation_invariant_checks':ablation_checks,
     'prefix_off_changed_vectors':sum(x['retrieval']!=y['retrieval']
         for x,y in zip(entries['eu'],entries['eu_no_repeated_caption_prefix'])),
     'max_caption_chars':max((len(e.safe_caption or '') for e in eus),default=0),
     'embedding_input_length_diagnostic':length_diag}
   full_rows.extend(rows);profiles.append(profile)
   if ablation_issues:leakage.extend([{'doc_id':doc_id,**z} for z in ablation_issues]);print('  [CROSS-FIELD-OVERLAP]',len(ablation_issues))
   print('  ✓ QA',len(rows),'Tables',len(doc.tables),'Base',len(hc),'EU',len(eus),
     'PageHit@1 base/eu',round(np.mean([x['arms']['base']['page_hit']['1'] for x in rows]),3),
     round(np.mean([x['arms']['eu']['page_hit']['1'] for x in rows]),3),flush=True)
  except Exception as e:
   failures.append({'doc_id':doc_id,'error':str(e),'traceback':traceback.format_exc()[-5000:]})
   print('  [FAILED]',type(e).__name__,str(e),flush=True)
  finally:
   dump_json(args.out_dir/'rows_checkpoint.json',full_rows)
   dump_json(args.out_dir/'documents_checkpoint.json',profiles)
   dump_json(args.out_dir/'failures.json',failures)
   dump_json(args.out_dir/'ablation_cross_field_overlaps.json',leakage)
   if 'doc' in locals():del doc
   if 'entries' in locals():del entries
   if 'corpora' in locals():del corpora
   if 'eus' in locals():del eus
   gc.collect()
 summary={'config':run_info,'n_expected_questions':sum(len(json.loads(x[1].read_text(encoding='utf-8'))) for x in all_pairs),
  'n_evaluated':len(full_rows),'n_documents':len(profiles),'failed_docs':failures,
  'document_profiles':profiles,
  'observed_input_sizes':{'min_pages':min((p['n_pages'] for p in profiles),default=None),
   'max_pages':max((p['n_pages'] for p in profiles),default=None),
   'max_table_rows':max((p['max_table_rows'] for p in profiles),default=None),
   'max_table_cols':max((p['max_table_cols'] for p in profiles),default=None),
   'note':'Successful observed sizes only: NOT official upper or lower support bounds.'},
  'ablation_cross_field_overlap_flags':len(leakage),
  'ablation_overlap_note':'An identical span in an independent field is a semantic overlap, not proof that the ablation code failed.',
  'token_length_diagnostic_note':'Evenly spaced sample of at most 256 retrieval vectors per PDF/arm; counts indicate risk of encoder truncation, not proven causal effect.'}
 if profiles:
  latency_fields=['parse_s','hybrid_chunk_s','eu_chunk_s','ablation_build_s','question_embedding_s']
  summary['latency']={
   'n_documents':len(profiles),
   'per_document_seconds':profiles,
   'stage_totals_seconds':{key:round(sum(p['timings'][key] for p in profiles),4) for key in latency_fields},
   'embedding_totals_seconds':{a:round(sum(p['timings']['embedding_per_arm_s'][a] for p in profiles),4) for a in ARMS},
   'ranking_totals_seconds':{a:round(sum(p['timings']['ranking_per_arm_s'][a] for p in profiles),4) for a in ARMS},
   'median_parse_seconds':round(float(np.median([p['timings']['parse_s'] for p in profiles])),4),
   'ranking_ms_per_question':{a:round(1000*sum(p['timings']['ranking_per_arm_s'][a] for p in profiles)/len(full_rows),4)
     for a in ARMS} if full_rows else {},
   'notes':['Shared PDF parse once per document, not counted five times in actual runtime',
      'Model loading, package installation, QA dataset construction excluded',
      'All timings sequential; repeated execution may benefit from cached models',
      'Do not claim isolated end-to-end latency comparisons from these stage timings']}
 if full_rows:
  summary['rates']={metric:{str(k) if k is not None else '1':{
   arm:round(float(np.mean([r['arms'][arm][metric][str(k)] if k is not None else r['arms'][arm][metric] for r in full_rows])),5)
    for arm in ARMS} for k in (K_LIST if metric!='evidence_hit_1' else (None,))}
    for metric in ('page_hit','table_hit','evidence_hit_1')}
  summary['paired']={metric+'@'+str(k):paired_stats(full_rows,'base','eu',metric,k)
   for metric,k in [('page_hit',1),('page_hit',5),('page_hit',10),('table_hit',1),('table_hit',5),('table_hit',10),('evidence_hit_1',None)]}
  summary['ablations']={arm:paired_stats(full_rows,'eu',arm,'evidence_hit_1')
    for arm in ARMS if arm.startswith('eu_no_')}
  summary['token_efficiency']=summarize_prompt_metrics(full_rows)
  summary['by_type']={t:{arm:round(float(np.mean([r['arms'][arm]['evidence_hit_1'] for r in full_rows if r['type']==t])),5)
    for arm in ARMS} for t in sorted({r['type'] for r in full_rows})}
  ctx=[r for r in full_rows if r['type']=='context_dependent']
  summary['context_distance_slices']={name:{'n':len(group),'base_evidence_hit_1':round(float(np.mean([r['arms']['base']['evidence_hit_1'] for r in group])),4) if group else None,
    'eu_evidence_hit_1':round(float(np.mean([r['arms']['eu']['evidence_hit_1'] for r in group])),4) if group else None}
   for name,group in {'within_300pt':[r for r in ctx if isinstance(r['dist_pt'],(float,int)) and r['dist_pt']<=300],
                      'beyond_300pt':[r for r in ctx if isinstance(r['dist_pt'],(float,int)) and r['dist_pt']>300],
                      'distance_missing':[r for r in ctx if not isinstance(r['dist_pt'],(float,int))]}.items()}
 dump_json(args.out_dir/'benchmark_summary.json',summary)
 print('\n=== RESULT ===',flush=True)
 print(json.dumps({'evaluated':len(full_rows),'expected':summary['n_expected_questions'],
   'failed_docs':len(failures),'rates':summary.get('rates')},indent=2,ensure_ascii=False),flush=True)
 if failures or len(full_rows)!=summary['n_expected_questions']:
  raise SystemExit('INCOMPLETE: Not publishable as final 90 PDF result; see failures.json and checkpoints')
 print('RUN COMPLETE — paired evaluation passed all preflight and document checks',flush=True)

if __name__=='__main__':main()
