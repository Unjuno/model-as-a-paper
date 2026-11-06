"""Download only bounded sample Parquet files; no corpus-wide snapshot download.

Requires the optional [hub] dependencies and internet. No source/PDF execution.
The sample is a wiring check, not a representative corpus or a benchmark.
"""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', required=True)
    p.add_argument('--revision', default='main')
    p.add_argument('--max-papers', type=int, default=256)
    p.add_argument('--max-download-mib', type=int, default=64)
    a = p.parse_args()
    if a.max_papers < 1 or a.max_download_mib < 1: p.error('limits must be positive')
    output = Path(a.output)
    if output.exists(): raise FileExistsError(output)
    from huggingface_hub import HfApi, hf_hub_download
    import pyarrow.parquet as pq
    from model_as_a_paper.data import DEFAULT_LICENSES
    repo = 'secemp9/arxiv-complete'; api = HfApi()
    revision = api.repo_info(repo, repo_type='dataset', revision=a.revision).sha
    files = [f for f in api.list_repo_tree(repo, repo_type='dataset', revision=revision,
                                         path_in_repo='sample', recursive=False)
             if getattr(f, 'path', '').endswith('.parquet')]
    if not files or sum(f.size for f in files) > a.max_download_mib * 1024**2:
        raise ValueError('sample missing or exceeds transfer bound; refusing broader download')
    output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output.open('x', encoding='utf-8') as dst:
        for entry in sorted(files, key=lambda f: f.path):
            path = hf_hub_download(repo, entry.path, repo_type='dataset', revision=revision)
            parquet = pq.ParquetFile(path)
            try:
                for batch in parquet.iter_batches(batch_size=32, columns=['paper_id', 'text', 'license']):
                    for row in batch.to_pylist():
                        text = row.get('text')
                        if row.get('license') not in DEFAULT_LICENSES or not isinstance(text, str) or len(text) < 1000:
                            continue
                        record = dict(paper_id=row['paper_id'], group_id=row['paper_id'], version=None,
                                      license=row['license'], kind='paper', text=text,
                                      source_repo=repo, source_revision=revision, source_file=entry.path,
                                      source_sha256=hashlib.sha256(text.encode()).hexdigest(),
                                      version_warning='assembled source has no verified arXiv revision')
                        dst.write(json.dumps(record, ensure_ascii=False)+'\n'); count += 1
                        if count >= a.max_papers: break
                    if count >= a.max_papers: break
            finally:
                parquet.close()
            if count >= a.max_papers: break
    if not count:
        output.unlink()
        raise ValueError('no eligible papers; check source schema/license selection')
    print(json.dumps({'papers':count, 'revision':revision, 'output':str(output),
                      'notice':'Sample only. Unknown source version is unsuitable for temporal correction claims.'}, indent=2))


if __name__ == '__main__': main()
