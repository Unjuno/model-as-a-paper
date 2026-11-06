"""Bounded ingestion and cloud configuration; never performs network calls."""
import importlib.util
import json
from pathlib import Path
import pytest


def test_import_row_preserves_provenance_not_fake_version():
    from model_as_a_paper.import_data import convert_arxiv_row
    row={'paper_id':'2401.12345','title':'Example','text':'\\begin{document}text','license':'https://creativecommons.org/licenses/by/4.0/'}
    r=convert_arxiv_row(row,'a'*40,'shard.parquet','b'*64)
    assert r['paper_id']=='2401.12345' and r['kind']=='paper'
    assert r['version'] is None and r['version_verified'] is False
    assert r['provenance']['dataset_revision']=='a'*40
    assert r['text']==row['text']
    with pytest.raises(ValueError): convert_arxiv_row({'paper_id':'x','text':''},'a'*40,'x','b'*64)
    with pytest.raises(ValueError): convert_arxiv_row(row,'main','x','b'*64)


def test_bounded_sample_selection():
    from model_as_a_paper.import_data import select_sample_files
    rows=[{'path':'sample/train-000.parquet','size':200},{'path':'pdf/full.parquet','size':10**12}]
    assert select_sample_files(rows,300)==['sample/train-000.parquet']
    with pytest.raises(ValueError): select_sample_files(rows,100)
    with pytest.raises(ValueError): select_sample_files([{'path':'sample/../source/x.parquet','size':1}],300)
    with pytest.raises(ValueError): select_sample_files([],300)


def test_cloud_request_is_bounded_and_paths_are_safe():
    from model_as_a_paper.cloud import validate_request
    plan=validate_request('pretrain-001','papers-v1',120,65536)
    assert plan['dataset_path']=='/vol/datasets/papers-v1'
    assert plan['run_path']=='/vol/runs/pretrain-001'
    assert plan['network_action']=='NONE' and not plan['billing_cap_guaranteed']
    for name in ('../x','/root','x/y','','x..y'):
        with pytest.raises(ValueError): validate_request(name,'ok',120,100)
    for seconds in (0,3001,-1):
        with pytest.raises(ValueError): validate_request('ok','data',seconds,100)


def test_single_writer_lease_contract():
    from model_as_a_paper.cloud import writer_lease
    class AtomicStore:
        def __init__(self): self.d={}
        def put(self,k,v,skip_if_exists=False):
            if skip_if_exists and k in self.d: return False
            self.d[k]=v; return True
        def get(self,k,default=None): return self.d.get(k,default)
        def pop(self,k): return self.d.pop(k)
    store=AtomicStore()
    with writer_lease(store,'run'):
        with pytest.raises(RuntimeError):
            with writer_lease(store,'run'): pass
        assert store.d
    assert not store.d
    with pytest.raises(ValueError):
        with writer_lease(store,'run'): raise ValueError('original failure')
    assert not store.d


def test_local_parquet_roundtrip(tmp_path):
    pa=pytest.importorskip('pyarrow'); pq=pytest.importorskip('pyarrow.parquet')
    from model_as_a_paper.import_data import import_parquet
    path=tmp_path/'data.parquet'
    pq.write_table(pa.Table.from_pylist([{'paper_id':'1234.5678','text':'a paper','license':'cc-by-4.0'}]),path)
    out=tmp_path/'papers.jsonl'
    result=import_parquet([str(path)],str(out),revision='a'*40,limit=1)
    assert result['written']==1
    assert json.loads(out.read_text())['paper_id']=='1234.5678'


def test_modal_module_can_import_without_remote_calls():
    pytest.importorskip('modal')
    path=Path(__file__).parents[1]/'modal_apps'/'train.py'
    spec=importlib.util.spec_from_file_location('map_modal_train',path)
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    assert module.app.name=='model-as-a-paper-training'
