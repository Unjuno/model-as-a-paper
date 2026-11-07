"""RED-first, network-free checks for bounded arXiv paper_text importer."""
from scripts.fetch_arxiv_papers import eligible_record, choose_shards


def test_eligible_paper_has_only_paper_origin_and_traceable_revision():
    row=dict(paper_id='2501.01234', title='Example', primary_category='math.AG',
             license='http://creativecommons.org/licenses/by/4.0/',
             text='\\documentclass{article}\\begin{document}Claim.\\end{document}' * 40)
    record=eligible_record(row, revision='0123456', shard='paper_text/train-0001.parquet',
                           prefixes=('math.',), min_chars=1000)
    assert record['kind'] == 'paper'
    assert record['paper_id'] == row['paper_id']
    assert record['source_revision'] == '0123456'
    assert record['version'] is None
    assert 'unversioned' in record['version_warning']
    assert record['text'] == row['text']


def test_unlicensed_or_wrong_category_or_stub_is_skipped():
    base=dict(paper_id='1234.00001',primary_category='cs.LG',
              license='http://creativecommons.org/licenses/by/4.0/',
              text='a scientific manuscript ' * 100)
    assert eligible_record(dict(base,license='http://arxiv.org/licenses/nonexclusive-distrib/1.0/'),
                           'x','shard',(),100) is None
    assert eligible_record(base,'x','shard',('math.',),100) is None
    assert eligible_record(dict(base,text='%auto-ignore'),'x','shard',(),1) is None
    assert eligible_record(dict(base,text='a\x00b'*100),'x','shard',(),1) is None


def test_shard_selection_deterministic_and_bounded():
    shards=[f'paper_text/train-{i:05d}-of-00050.parquet' for i in range(50)]
    assert choose_shards(shards,seed=17,limit=3)==choose_shards(shards,seed=17,limit=3)
    assert len(choose_shards(shards,seed=17,limit=3))==3
    assert len(set(choose_shards(shards,seed=17,limit=3)))==3


def test_setup_supports_bounded_paper_text_mode(tmp_path, monkeypatch, capsys):
    import json
    from pathlib import Path
    from scripts import setup_training
    from test_training import build, config
    build(tmp_path)
    source=(tmp_path/'data.jsonl').read_text()
    cfg=tmp_path/'cfg.json'; cfg.write_text(json.dumps(config()))
    commands=[]
    def fake_run(cmd, check, **kwargs):
        commands.append(cmd)
        Path(cmd[cmd.index('--output')+1]).write_text(source)
    monkeypatch.setattr(setup_training.subprocess,'run',fake_run)
    out=tmp_path/'new'
    setup_training.main(['--source','paper-text','--output',str(out),
                         '--config',str(cfg),'--vocab-size','300',
                         '--max-papers','20','--max-shards','2',
                         '--min-chars','1','--allow-test-data'])
    result=json.loads(capsys.readouterr().out)
    assert result['trainable'] is True
    assert len(commands)==1
    assert 'fetch_arxiv_papers.py' in ' '.join(commands[0])
    assert '--max-shards' in commands[0]


def test_remote_import_contract_with_bounded_fake_arrow(tmp_path, monkeypatch):
    """No HTTP: exercises pinned paths, schema, row groups, filtering and output."""
    import json
    import sys
    import types
    from contextlib import nullcontext
    from scripts.fetch_arxiv_papers import fetch
    import huggingface_hub
    desired=dict(paper_id='2501.01234', title='Sample theorem', primary_category='math.AG',
                 license='http://creativecommons.org/licenses/by/4.0/',
                 text='A mathematical paper with proof. ' * 100)
    rejected=dict(desired, paper_id='2501.01235',license='unlicensed')
    visited=[]
    class FakeAPI:
        def repo_info(self,*args,**kwargs):return types.SimpleNamespace(sha='a'*40)
        def list_repo_tree(self,*args,**kwargs):
            return [types.SimpleNamespace(path='paper_text/train-00000-of-00050.parquet')]
    class FakeFS:
        def open(self,path,mode):
            visited.append(path)
            return nullcontext('fake parq')
    class FakeBatch:
        def to_pylist(self):return [desired,rejected]
    class FakeColumn:
        total_compressed_size=128
    class FakeRowGroup:
        def column(self,index):return FakeColumn()
    class FakeMetadata:
        num_row_groups=1
        def row_group(self,index):return FakeRowGroup()
    class FakeParquetFile:
        def __init__(self,handle):
            self.schema_arrow=types.SimpleNamespace(names=['paper_id','text','license','primary_category','title'])
            self.metadata=FakeMetadata()
        def iter_batches(self,**kwargs):
            assert kwargs['row_groups'] == [0]
            yield FakeBatch()
    parq = types.ModuleType('pyarrow.parquet')
    parq.ParquetFile=FakeParquetFile
    arrow = types.ModuleType('pyarrow');arrow.parquet=parq
    monkeypatch.setitem(sys.modules,'pyarrow',arrow)
    monkeypatch.setitem(sys.modules,'pyarrow.parquet',parq)
    monkeypatch.setattr(huggingface_hub,'HfApi',FakeAPI)
    monkeypatch.setattr(huggingface_hub,'HfFileSystem',FakeFS)
    out=tmp_path/'paper.jsonl'
    result=fetch(out,max_papers=2,max_shards=1,max_output_mib=1,
                 max_scan_mib=1,category_prefixes=('math.',))
    docs=[json.loads(s) for s in out.read_text().splitlines()]
    assert len(docs)==1
    assert result['papers']==1
    assert docs[0]['paper_id']=='2501.01234'
    assert '@'+'a'*40+'/' in visited[0]
    assert result['scanned_compressed_bytes_estimate']>0
