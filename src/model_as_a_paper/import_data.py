"""Bounded local ingestion. Never downloads the entire multi-terabyte repository."""
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import tempfile

from .io import atomic_json, file_sha256

REPO = 'secemp9/arxiv-complete'


def _revision(value):
    if not isinstance(value,str) or not re.fullmatch(r'[0-9a-f]{40}',value):
        raise ValueError('An immutable 40-character Hub commit SHA is required')


def convert_arxiv_row(row,revision,shard_name,shard_sha256):
    _revision(revision)
    pid=row.get('paper_id'); text=row.get('text')
    if not isinstance(pid,str) or not pid.strip() or not isinstance(text,str) or not text.strip():
        raise ValueError('paper_text row must have nonempty paper_id and text')
    # paper_text sources are unversioned: never infer v1 or newest version.
    return dict(id=f'arxiv:{pid}',paper_id=pid,group_id=pid,kind='paper',
                source=f'https://arxiv.org/abs/{pid}',version=None,version_verified=False,
                title=row.get('title'),license=row.get('license'),text=text,
                provenance=dict(dataset=REPO,dataset_revision=revision,
                                shard=shard_name,shard_sha256=shard_sha256,
                                source_text_sha256=row.get('text_sha256')))


def select_sample_files(entries,max_bytes):
    if type(max_bytes) is not int or max_bytes<=0: raise ValueError('Download cap must be positive')
    selected=[]; total=0
    for entry in entries:
        name=entry['path']; parts=PurePosixPath(name).parts
        if not name.startswith('sample/'): continue
        if '..' in parts or PurePosixPath(name).is_absolute(): raise ValueError('Unsafe sample path')
        if not name.endswith('.parquet'): continue
        size=entry.get('size')
        if type(size) is not int or size<0: raise ValueError('Missing sample file size')
        selected.append(name); total+=size
    if not selected or total>max_bytes: raise ValueError('Sample missing or exceeds download cap')
    return sorted(selected)


def fetch_sample(out_dir,*,revision=None,max_download_bytes=50_000_000):
    from huggingface_hub import HfApi, hf_hub_download
    out=Path(out_dir)
    if out.exists(): raise FileExistsError(f'Refusing to overwrite {out}')
    api=HfApi(); info=api.dataset_info(REPO,revision=revision)
    sha=info.sha; _revision(sha)
    entries=[dict(path=r.path,size=r.size) for r in api.list_repo_tree(REPO,repo_type='dataset',
                 revision=sha,path_in_repo='sample',recursive=True) if hasattr(r,'size')]
    selected=select_sample_files(entries,max_download_bytes)
    out.parent.mkdir(parents=True,exist_ok=True)
    staging=Path(tempfile.mkdtemp(prefix='.sample-',dir=out.parent))
    try:
        files=[]; actual=0
        for name in selected:
            cached=Path(hf_hub_download(REPO,name,repo_type='dataset',revision=sha))
            actual+=cached.stat().st_size
            if actual>max_download_bytes: raise ValueError('Actual transfer payload exceeds cap')
            target=staging/Path(name).name; shutil.copyfile(cached,target)
            files.append(dict(path=target.name,bytes=target.stat().st_size,sha256=file_sha256(target)))
        manifest=dict(dataset=REPO,revision=sha,files=files,bytes=actual,
                      warning='Sample is a pipeline check, not a sufficient scientific training set')
        atomic_json(staging/'download.json',manifest); os.replace(staging,out)
        return manifest
    except BaseException:
        shutil.rmtree(staging,ignore_errors=True); raise


def import_parquet(files,out_file,*,revision,limit=1000):
    _revision(revision)
    if type(limit) is not int or limit<=0: raise ValueError('limit must be positive')
    import pyarrow.parquet as pq
    out=Path(out_file)
    if out.exists(): raise FileExistsError(f'Refusing to overwrite {out}')
    out.parent.mkdir(parents=True,exist_ok=True)
    fd,temp=tempfile.mkstemp(prefix='.import-',dir=out.parent)
    written=0; sources=[]
    try:
        with os.fdopen(fd,'w',encoding='utf-8') as handle:
            for filename in files:
                path=Path(filename); digest=file_sha256(path); sources.append(dict(path=path.name,sha256=digest))
                with pq.ParquetFile(path) as pf:
                    names=pf.schema_arrow.names
                    if not {'paper_id','text'}.issubset(names): raise ValueError('Expected paper_text/sample schema')
                    columns=[n for n in ('paper_id','text','title','license','text_sha256') if n in names]
                    for batch in pf.iter_batches(batch_size=32,columns=columns):
                        for row in batch.to_pylist():
                            result=convert_arxiv_row(row,revision,path.name,digest)
                            handle.write(json.dumps(result,ensure_ascii=False,allow_nan=False)+'\n'); written+=1
                            if written>=limit: break
                        if written>=limit: break
                if written>=limit: break
            if not written: raise ValueError('No paper rows imported')
            handle.flush(); os.fsync(handle.fileno())
        os.replace(temp,out)
        return dict(written=written,revision=revision,sources=sources,output_sha256=file_sha256(out),
                    note='Metadata records provenance, not legal approval or scientific validation')
    except BaseException:
        if os.path.exists(temp): os.unlink(temp)
        raise
