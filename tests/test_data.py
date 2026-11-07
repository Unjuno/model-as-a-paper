import json
import numpy as np
import pytest
from model_as_a_paper.tokenizer import ByteBPE
from model_as_a_paper.data import prepare, Corpus, WindowDataset, StatefulStream


def rows(n=15):
    return [dict(paper_id=f'TEST-{i}', group_id=f'family-{i}', version='fixture',
                 license='TEST-ONLY', source_kind='test_fixture',
                 text=f'Paper {i}. Conditions and counterexamples. \\alpha + x = y. ' * 3)
            for i in range(n)]


def write_rows(tmp_path, items):
    p = tmp_path / 'input.jsonl'
    p.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in items))
    return p


def test_bpe_utf8_math_roundtrip_and_saved_identity(tmp_path):
    text = '条件 x ≤ 3。\\frac{α}{β} = 2\n\t A  B <sep>🙂'
    tok = ByteBPE.train([text * 5], vocab_size=300, min_frequency=2)
    assert tok.decode(tok.encode(text)) == text
    assert tok.sep_id not in tok.encode('<sep>')
    tok.save(tmp_path / 'tok.json')
    other = ByteBPE.load(tmp_path / 'tok.json')
    assert other.fingerprint == tok.fingerprint
    assert other.encode(text) == tok.encode(text)


def test_bpe_merges_reduce_positions_and_deterministic():
    text = 'abcdefgh abcdefgh abcdefgh ' * 40
    a = ByteBPE.train([text], vocab_size=280)
    b = ByteBPE.train([text], vocab_size=280)
    assert len(a.encode(text)) < len(text.encode()) / 3
    assert a.fingerprint == b.fingerprint


def test_data_refuses_fixtures_by_default_and_unverified_corrections(tmp_path):
    p = write_rows(tmp_path, rows())
    with pytest.raises(ValueError): prepare(p, tmp_path/'a', vocab_size=300)
    r = rows(1)[0]; r.pop('text')
    r.update(kind='correction', prompt='question', answer='answer', verified=False)
    p = write_rows(tmp_path, [r])
    with pytest.raises(ValueError):
        prepare(p, tmp_path/'b', vocab_size=300, allow_test_data=True, min_chars=1)


def test_group_and_duplicate_split_isolation_and_next_token(tmp_path):
    data = rows()
    data.append(dict(data[0], paper_id='other', group_id='different'))
    data.append(dict(data[1], paper_id='TEST-1v2', group_id='changed', text=data[1]['text']+'revision'))
    p = write_rows(tmp_path, data)
    prepare(p, tmp_path/'corpus', vocab_size=300, allow_test_data=True, min_chars=1)
    c = Corpus(tmp_path/'corpus')
    seen = {}
    for r in c.records:
        for key in (r['group_id'], r['paper_id'].split('v')[0]):
            assert key not in seen or seen[key] == r['split']
            seen[key] = r['split']
    d = WindowDataset(c, 'train', 16)
    x, y, sizes = d[0]
    start, end = d.windows[0]
    assert np.array_equal(x[:end-start-1], c.tokens[start:end-1])
    assert np.array_equal(y[:end-start-1], c.tokens[start+1:end])
    assert (y[end-start-1:] == -100).all()
    assert sizes[y == -100].sum() == 0


def test_response_only_mask_and_no_empty_target_windows(tmp_path):
    data = rows(1)
    tok = ByteBPE.train([data[0]['text']], vocab_size=300)
    tok.save(tmp_path/'tok.json')
    r = data[0]; r.pop('text')
    r.update(kind='correction', prompt='Question: ' * 50, answer='Yes.', verified=True,
             evidence=[{'paper_id':'TEST-0', 'version':'fixture', 'span':'test only'}],
             verification='software fixture, not a scientific validation')
    p = write_rows(tmp_path, [r])
    prepare(p, tmp_path/'sft', tokenizer_path=tmp_path/'tok.json',
            allow_test_data=True, min_chars=1, split='train')
    c = Corpus(tmp_path/'sft'); d = WindowDataset(c, 'train', 4096)
    actual = []
    for i in range(len(d)):
        x, y, _ = d[i]
        assert (y != -100).any()
        actual.extend(y[y != -100].tolist())
    assert actual == tok.encode('Yes.') + [tok.eos_id]


def test_tokenizer_uses_only_train_records(tmp_path):
    data = rows()
    p = write_rows(tmp_path, data)
    prepare(p, tmp_path/'corpus', vocab_size=300, allow_test_data=True, min_chars=1)
    c = Corpus(tmp_path/'corpus')
    train_ids = set(c.manifest['tokenizer_training_paper_ids'])
    eval_ids = {r['paper_id'] for r in c.records if r['split'] != 'train'}
    assert train_ids.isdisjoint(eval_ids)
    assert train_ids


def test_stream_resume_exact_and_corruption_rejected(tmp_path):
    p = write_rows(tmp_path, rows())
    prepare(p, tmp_path/'corpus', vocab_size=300, allow_test_data=True, min_chars=1)
    c = Corpus(tmp_path/'corpus'); d = WindowDataset(c, 'train', 16)
    a = StatefulStream(d, seed=9)
    for _ in range(3): a.next()
    state = a.state_dict()
    expected = [a.next()[0] for _ in range(5)]
    b = StatefulStream(d, seed=100); b.load_state_dict(state)
    for x in expected: assert np.array_equal(x, b.next()[0])
    with (tmp_path/'corpus'/'tokens.bin').open('r+b') as f: f.write(b'\x00\x00')
    with pytest.raises(ValueError, match='hash'):
        Corpus(tmp_path/'corpus')


def test_preparation_never_overwrites_existing_data(tmp_path):
    p = write_rows(tmp_path, rows())
    out = tmp_path/'existing'; out.mkdir(); (out/'precious.txt').write_text('keep')
    with pytest.raises(FileExistsError):
        prepare(p, out, vocab_size=300, allow_test_data=True, min_chars=1)
    assert (out/'precious.txt').read_text() == 'keep'
