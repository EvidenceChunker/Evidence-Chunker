#!/usr/bin/env python3
'Build audited QA copies with verified table indices and source hash checks.'
import argparse
import collections
import copy
import hashlib
import json
from pathlib import Path


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--qa-dir", required=True, type=Path, help="Directory containing original *_qa.json files")
    p.add_argument("--bundle-dir", default=Path(__file__).resolve().parent, type=Path,
                   help="Directory containing verified_qa_mapping.json and pdf_audit_summary.json")
    p.add_argument("--out-dir", required=True, type=Path, help="New output directory (must not be original QA directory)")
    args = p.parse_args()
    original_dir = args.qa_dir.resolve()
    out_dir = args.out_dir.resolve()
    if out_dir == original_dir or original_dir in out_dir.parents:
        raise SystemExit("Output must not be inside the original QA directory")
    mapping_file = args.bundle_dir / 'verified_qa_mapping.json'
    docs_file = args.bundle_dir / 'pdf_audit_summary.json'
    mapping = json.loads(mapping_file.read_text(encoding='utf-8'))
    docs = json.loads(docs_file.read_text(encoding='utf-8'))
    by_doc = collections.defaultdict(dict)
    for r in mapping['verified_questions']:
        d, qid = r['doc_id'], r['qid']
        if qid in by_doc[d]:
            raise ValueError(f'Duplicate qid in manifest: {qid}')
        by_doc[d][qid] = r
    if len(by_doc) != 90:
        raise ValueError(f'Expected 90 documents in verified manifest; found {len(by_doc)}')
    if out_dir.exists() and any(out_dir.iterdir()):
        raise ValueError('Output directory must be empty for a reproducible fresh build')
    checked = []
    for doc in docs:
        if doc['status'] != 'parsed':
            raise ValueError(f'Input was not audited successfully: {doc["doc_id"]}')
        doc_id = doc['doc_id']
        src = original_dir / f'{doc_id}_qa.json'
        if not src.is_file():
            raise FileNotFoundError(src)
        observed_hash = sha256(src)
        if observed_hash != doc['qa_sha256']:
            raise ValueError(f'Original QA checksum changed: {src} expected={doc["qa_sha256"]} found={observed_hash}')
        original = json.loads(src.read_text(encoding='utf-8'))
        if not isinstance(original, list):
            raise ValueError(f'Unexpected QA JSON shape: {src}')
        qids = [q.get('qid') for q in original]
        if len(qids) != len(set(qids)):
            raise ValueError(f'Duplicate qids in original QA: {doc_id}')
        missing = set(by_doc[doc_id]) - set(qids)
        if missing:
            raise ValueError(f'QIDs missing in source {doc_id}: {sorted(missing)[:3]}')
        if len(original) != doc['n_qa']:
            raise ValueError(f'QA count changed: {doc_id}')
        checked.append((doc, original))
    out_dir.mkdir(parents=True, exist_ok=True)
    output_docs=[]
    verified_total=0
    changed_total=0
    for doc, originals in checked:
        doc_id = doc['doc_id']
        keep = by_doc[doc_id]
        filtered=[]
        for src_qa in originals:
            r=keep.get(src_qa.get('qid'))
            if r is None:
                continue
            qa=copy.deepcopy(src_qa)
            old_index=(qa.get('meta') or {}).get('table_index')
            if old_index != r['old_table_index'] or qa.get('page') != r['page'] or qa.get('type') != r['type']:
                raise ValueError('Audit metadata mismatch: '+str(qa.get('qid')))
            qa.setdefault('meta', {})['table_index'] = r['new_table_index']
            if old_index != r['new_table_index']:
                changed_total += 1
            filtered.append(qa)
        if len(filtered) != len(keep):
            raise ValueError('Selection count mismatch: '+doc_id)
        verified_total += len(filtered)
        dst=out_dir / f'{doc_id}_qa.json'
        dst.write_text(json.dumps(filtered, ensure_ascii=False, indent=2), encoding='utf-8')
        output_docs.append({'doc_id':doc_id,'source_sha256':doc['qa_sha256'],
                            'output_sha256':sha256(dst), 'input_count':len(originals),
                            'verified_count':len(filtered)})
    assert verified_total==len(mapping['verified_questions'])==2239
    assert changed_total==mapping['changed_table_index_count']==602
    summary={'dataset_name':'QA_v2_verified',
             'source_dataset':'Original 90-PDF QA (unchanged)',
             'docling_version':mapping['docling_version'],
             'docling_core_version':mapping['docling_core_version'],
             'audit_sha256':mapping['audit_details_sha256'],
             'original_qa_count':sum(x['input_count'] for x in output_docs),
             'verified_qa_count':verified_total,'remapped_table_index_count':changed_total,
             'n_docs':len(output_docs),'documents':output_docs,
             'note':'Filtered solely on source-only Docling audit status; no baseline/EU results consulted. The only changed QA field is meta.table_index in these NEW evaluation copies.'}
    (out_dir / '_verified_dataset_manifest.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({k:v for k,v in summary.items() if k!='documents'},ensure_ascii=False,indent=2))

if __name__=='__main__':
    main()
