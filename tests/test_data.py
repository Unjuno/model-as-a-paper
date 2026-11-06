import json
from pathlib import Path
import pytest
import torch
from helpers import ByteCodec, encoded_fixture, paper, write_rows


def api():
    try:
        from model_as_a_paper.corpus import prepare_corpus
        from model_as_a_paper.data import BlockDataset, BlockStream, EncodedCorpus
    except ImportError:
        pytest.fail('Data implementation is missing')
    return prepare_corpus, BlockDataset, BlockStream, EncodedCorpus


def test_prepare_requires_provenance_and_filters_unknown_license(tmp_path):
    prepare, *_ = api()
    rows = [paper(1), paper(2, split='validation'), paper(3, split='test')]
    rows.append({**paper(4), 'license':None})
    src = write_rows(tmp_path/'input.jsonl', rows)
    result = prepare(src, tmp_path/'corpus', min_chars=1, allow_test_data=True)
    assert result['counts'] == {'train':1,'validation':1,'test':1}
    assert result['rejected']['license'] == 1
    bad = paper(5); bad.pop('source')
    src2 = write_rows(tmp_path/'bad.jsonl', [bad])
    with pytest.raises(ValueError, match='source'):
        prepare(src2, tmp_path/'bad', min_chars=1, allow_test_data=True)


def test_split_leakage_and_exact_duplicate_conflicts_rejected(tmp_path):
    prepare, *_ = api()
    rows = [paper(1), {**paper(2), 'paper_id':'paper-1', 'split':'validation'}]
    with pytest.raises(ValueError, match='split|group|family'):
        prepare(write_rows(tmp_path/'bad.jsonl', rows), tmp_path/'a', min_chars=1, allow_test_data=True)
    rows = [paper(1, 'duplicate text'), paper(2, 'duplicate text', split='validation')]
    with pytest.raises(ValueError, match='duplicate'):
        prepare(write_rows(tmp_path/'bad2.jsonl', rows), tmp_path/'b', min_chars=1, allow_test_data=True)


def test_unverified_correction_cannot_enter_training(tmp_path):
    prepare, *_ = api()
    r = paper(1); r.update(kind='correction', prompt='Q', response='A')
    with pytest.raises(ValueError, match='verif'):
        prepare(write_rows(tmp_path/'bad.jsonl', [r]), tmp_path/'a', min_chars=1, allow_test_data=True)


def test_test_fixture_forbidden_by_default(tmp_path):
    prepare, *_ = api()
    with pytest.raises(ValueError, match='test'):
        prepare(write_rows(tmp_path/'bad.jsonl', [paper(1)]), tmp_path/'a', min_chars=1)


def test_shift_labels_no_cross_document_and_exact_bytes(tmp_path):
    _, Blocks, _, Corpus = api()
    corpus = Corpus(encoded_fixture(tmp_path/'encoded'))
    blocks = Blocks(corpus, 'train', 16)
    total_bytes, supervised = 0, 0
    for i in range(len(blocks)):
        x, y, byte_n = blocks[i]
        valid = y != -100
        total_bytes += int(byte_n[valid].sum())
        supervised += int(valid.sum())
        n = blocks.blocks[i][1]
        assert (y[:n-1][y[:n-1] != -100] == x[1:n][y[:n-1] != -100]).all()
        assert int((x == 1).sum()) <= 1
    assert total_bytes == corpus.manifest['splits']['train']['text_bytes']
    assert supervised == corpus.manifest['splits']['train']['supervised_tokens']


def test_sft_only_answers_are_supervised(tmp_path):
    _, Blocks, _, Corpus = api()
    c = Corpus(encoded_fixture(tmp_path/'encoded', kind='sft'))
    ds = Blocks(c, 'train', 128)
    x, y, _ = ds[0]
    sep = int((x == 3).nonzero()[0])
    assert (y[:sep] == -100).all()
    assert y[sep] >= 4
    assert y[y != -100][-1] == 2


def test_stream_resume_preserves_permutation_and_cursor(tmp_path):
    _, Blocks, Stream, Corpus = api()
    ds = Blocks(Corpus(encoded_fixture(tmp_path/'encoded')), 'train', 16)
    a = Stream(ds, seed=10)
    a.batch(3); state = a.state_dict()
    expected = a.batch(4)
    b = Stream(ds, seed=10); b.load_state_dict(state)
    actual = b.batch(4)
    for x, y in zip(expected, actual): torch.testing.assert_close(x, y, rtol=0, atol=0)


def test_corrupted_encoded_bytes_rejected(tmp_path):
    _, _, _, Corpus = api()
    p = encoded_fixture(tmp_path/'encoded')
    with open(p/'train.ids.bin','ab') as f: f.write(b'bad')
    with pytest.raises(ValueError, match='hash|integrity'):
        Corpus(p)


def test_byte_bpe_round_trip_and_train_only(tmp_path):
    pytest.importorskip('tokenizers')
    from model_as_a_paper.corpus import prepare_corpus
    from model_as_a_paper.tokenization import train_tokenizer, Codec
    rows = [paper(i, text=('alpha beta gamma delta. x+y=0. '*20)+str(i)) for i in range(10)]
    rows += [paper('val', split='validation'), paper('test', split='test')]
    c = tmp_path/'corpus'
    prepare_corpus(write_rows(tmp_path/'in.jsonl', rows), c, min_chars=1, allow_test_data=True)
    path = tmp_path/'tokenizer.json'
    train_tokenizer(c, path, vocab_size=300, min_frequency=1)
    tok = Codec(path)
    text = '日本語 🧪 \\frac{α}{β}\n  x ≤ 1\t\r\n'
    ids = tok.encode(text)
    assert tok.decode(ids) == text
    assert sum(tok.byte_lengths[i] for i in ids) == len(text.encode('utf-8'))
    meta = json.loads(path.with_suffix('.meta.json').read_text())
    assert meta['trained_split'] == 'train'


def test_supervised_prompt_is_never_silently_lost(tmp_path):
    _, Blocks, _, Corpus = api()
    c=Corpus(encoded_fixture(tmp_path/'d',kind='sft'))
    with pytest.raises(ValueError,match='context'):
        Blocks(c,'train',16)
