"""One explicit, bounded path from paper JSONL to preflighted training corpus.

Default downloads only the 26 MB arxiv-complete sample, not all arXiv.
No cloud account operation, model training, GPU allocation, or charge occurs.
"""
import argparse
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import subprocess
import sys

from model_as_a_paper.data import prepare
from model_as_a_paper.readiness import preflight


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', help='existing paper-only JSONL; omit to fetch selected HF source')
    p.add_argument('--source', choices=['sample','paper-text'], default='sample')
    p.add_argument('--output', default='data/papers')
    p.add_argument('--config', default='configs/map44.json')
    p.add_argument('--max-papers', type=int, default=256)
    p.add_argument('--max-download-mib', type=int, default=64)
    p.add_argument('--max-shards', type=int, default=3)
    p.add_argument('--max-scan-mib', type=int, default=256)
    p.add_argument('--max-output-mib', type=int, default=192)
    p.add_argument('--category-prefix', action='append', default=[])
    p.add_argument('--min-chars', type=int, default=1000)
    p.add_argument('--vocab-size', type=int, default=16384)
    p.add_argument('--allow-test-data', action='store_true')
    a = p.parse_args(argv)
    output = Path(a.output)
    if output.exists(): p.error('output already exists; refusing overwrite')
    if min(a.max_papers, a.max_download_mib, a.max_shards, a.max_scan_mib, a.max_output_mib) <= 0:
        p.error('positive limits required')
    src = Path(a.input) if a.input else output.with_suffix('.jsonl')
    if not a.input:
        if src.exists(): p.error('download target already exists; refusing overwrite')
        try:
            # Exactly one JSON object goes to stdout. The fetcher has its own
            # progress/JSON output which must not corrupt automation consumers.
            if a.source == 'sample':
                fetch_args = [sys.executable, str(Path(__file__).parent/'fetch_arxiv_sample.py'),
                              '--output',str(src), '--max-papers',str(a.max_papers),
                              '--max-download-mib',str(a.max_download_mib)]
            else:
                fetch_args = [sys.executable, str(Path(__file__).parent/'fetch_arxiv_papers.py'),
                              '--output',str(src), '--max-papers',str(a.max_papers),
                              '--max-shards',str(a.max_shards),
                              '--max-scan-mib',str(a.max_scan_mib),
                              '--max-output-mib',str(a.max_output_mib),
                              '--min-chars',str(a.min_chars)]
                for prefix in a.category_prefix:
                    fetch_args.extend(['--category-prefix',prefix])
            with redirect_stdout(io.StringIO()):
                fetched = subprocess.run(fetch_args, capture_output=True, text=True, check=True)
        except subprocess.CalledProcessError as e:
            p.error(f'sample download failed (exit={e.returncode}): {(e.stderr or "")[-400:]}')
    else:
        if not src.exists(): p.error('input JSONL not found')
    manifest = prepare(src, output, vocab_size=a.vocab_size, min_chars=a.min_chars,
                       allow_test_data=a.allow_test_data)
    cfg = json.loads(Path(a.config).read_text())
    # If an explicitly supplied config differs from the tokenizer capacity,
    # reject the prepared corpus rather than silently launching an unusable run.
    result = preflight(cfg, output, allow_test_data=a.allow_test_data)
    result['source_jsonl'] = str(src)
    result['source_mode'] = 'local-jsonl' if a.input else a.source
    result['prepared_corpus'] = str(output)
    result['actions_executed'] = 'CPU data preparation and local preflight only; no CUDA, Modal or GPU jobs'
    result['suggested_next_command'] = (
        f'python -m model_as_a_paper.cli bench --config {a.config} '
        f'--data {output} --steps 2 --device cuda'
    )
    result['warning'] = ('HF sample is for pipeline verification; it is not enough for '
                         'a scientifically capable model. License and provenance review is mandatory.')
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
