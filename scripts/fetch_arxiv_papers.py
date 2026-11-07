"""Bounded, version-pinned remote extraction from arxiv-complete/paper_text.

Uses Arrow row-group reads via HfFileSystem: it must NOT download the whole
70 GB paper_text config or 16 TB dataset snapshot. External fetch is opt-in.
Source is assembled and NOT versioned; unsuitable for temporal correction QA.
Actual HTTP transfer can exceed Parquet compressed-byte estimates (metadata,
range requests, caching), so max-scan-mib is a conservative scan-planning
limit, not an enforced network/billing cap.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import uuid

from model_as_a_paper.data import DEFAULT_LICENSES, canonical_id


def choose_shards(paths, seed=17, limit=3):
    if type(limit) is not int or limit < 1: raise ValueError('positive shard count required')
    available = sorted(set(paths))
    if not available: raise ValueError('no parquet shards found')
    return random.Random(seed).sample(available, min(limit, len(available)))


def eligible_record(row, revision, shard, prefixes=(), min_chars=1000):
    """Keep text from scientific papers with the requested declared licences."""
    if not isinstance(row, dict) or row.get('license') not in DEFAULT_LICENSES:
        return None
    pid, body = row.get('paper_id'), row.get('text')
    if not isinstance(pid, str) or not pid.strip() or not isinstance(body, str):
        return None
    category = row.get('primary_category') or ''
    if prefixes and not any(category.startswith(prefix) for prefix in prefixes):
        return None
    if len(body) < min_chars or '\x00' in body:
        return None
    if body.strip() == '%auto-ignore' or ('\\includepdf' in body and len(body) < 1000):
        return None
    sha = hashlib.sha256(body.encode('utf-8')).hexdigest()
    return {
        'paper_id': pid,
        'group_id': canonical_id(pid),
        'version': None,
        'license': row['license'],
        'title': row.get('title'),
        'primary_category': category,
        'kind': 'paper',
        'text': body,
        'source_repo': 'secemp9/arxiv-complete',
        'source_revision': revision,
        'source_file': shard,
        'source_sha256': sha,
        'version_warning': 'unversioned assembled TeX: do not use as pre-correction temporal ground truth',
    }


def fetch(output, *, max_papers=4096, max_shards=3, max_output_mib=192,
          max_scan_mib=256, min_chars=1000, category_prefixes=(), seed=17):
    if any(type(n) is not int or n < 1 for n in
           (max_papers, max_shards, max_output_mib, max_scan_mib, min_chars)):
        raise ValueError('all resource limits must be positive integers')
    output = Path(output)
    if output.exists(): raise FileExistsError(output)
    # Import lazily so plain CPU training/tests do not need network packages.
    from huggingface_hub import HfApi, HfFileSystem
    import pyarrow.parquet as pq
    repo = 'secemp9/arxiv-complete'
    revision = HfApi().repo_info(repo, repo_type='dataset').sha
    entries = [f.path for f in HfApi().list_repo_tree(
        repo, path_in_repo='paper_text', repo_type='dataset',
        revision=revision, recursive=False) if getattr(f, 'path', '').endswith('.parquet')]
    shards = choose_shards(entries, seed, max_shards)
    fs = HfFileSystem()
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_name(output.name + f'.part-{uuid.uuid4().hex}')
    counts, total_bytes, scan_bytes, rowgroups = 0, 0, 0, 0
    scanned_shards = []
    max_out = max_output_mib * 1024**2
    max_scan = max_scan_mib * 1024**2
    stop = False
    try:
        with tmp.open('x', encoding='utf-8') as sink:
            for shard in shards:
                remote = f'datasets/{repo}@{revision}/{shard}'
                with fs.open(remote, 'rb') as raw:
                    pf = pq.ParquetFile(raw)
                    columns = ['paper_id','text','license','primary_category','title']
                    if not set(columns).issubset(pf.schema_arrow.names):
                        raise ValueError(f'paper_text schema changed: {shard}')
                    column_index = [pf.schema_arrow.names.index(c) for c in columns]
                    for i in range(pf.metadata.num_row_groups):
                        metadata = pf.metadata.row_group(i)
                        est_bytes = sum(metadata.column(c).total_compressed_size for c in column_index)
                        if scan_bytes + est_bytes > max_scan:
                            stop = True
                            break
                        scan_bytes += est_bytes
                        rowgroups += 1
                        for batch in pf.iter_batches(batch_size=64, row_groups=[i], columns=columns):
                            for row in batch.to_pylist():
                                record = eligible_record(row, revision, shard,
                                                         category_prefixes, min_chars)
                                if record is None: continue
                                encoded = (json.dumps(record, ensure_ascii=False)+'\n').encode('utf-8')
                                if total_bytes + len(encoded) > max_out:
                                    stop = True
                                    break
                                sink.write(encoded.decode('utf-8'))
                                counts += 1
                                total_bytes += len(encoded)
                                if counts >= max_papers:
                                    stop = True
                                    break
                            if stop: break
                        if stop: break
                scanned_shards.append(shard)
                if stop: break
            sink.flush(); os.fsync(sink.fileno())
        if counts < 1:
            raise ValueError('no eligible paper text found within bounds; adjust scan/categories')
        os.replace(tmp, output)
    finally:
        if tmp.exists(): tmp.unlink()
    return {
        'papers':counts, 'output_bytes':total_bytes, 'scanned_compressed_bytes_estimate':scan_bytes,
        'scanned_row_groups':rowgroups, 'shards_visited':scanned_shards,
        'dataset_revision':revision, 'output':str(output),
        'notice':'Bounded unversioned TeX with mixed source licences; not ground truth for temporal corrections.',
    }


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',required=True)
    p.add_argument('--max-papers',type=int,default=4096)
    p.add_argument('--max-shards',type=int,default=3)
    p.add_argument('--max-output-mib',type=int,default=192)
    p.add_argument('--max-scan-mib',type=int,default=256)
    p.add_argument('--min-chars',type=int,default=1000)
    p.add_argument('--category-prefix',action='append',default=[])
    p.add_argument('--seed',type=int,default=17)
    a=p.parse_args(argv)
    result=fetch(a.output,max_papers=a.max_papers,max_shards=a.max_shards,
                 max_output_mib=a.max_output_mib,max_scan_mib=a.max_scan_mib,
                 min_chars=a.min_chars,category_prefixes=tuple(a.category_prefix),seed=a.seed)
    print(json.dumps(result,ensure_ascii=False,indent=2))
    return 0


if __name__=='__main__':
    raise SystemExit(main())
