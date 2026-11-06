"""Synthetic software-only integration demonstration; never scientific training data."""
import json
from pathlib import Path
import torch
from .data import Corpus, WindowDataset, prepare
from .model import ModelConfig, PaperLM
from .training import TrainConfig, Trainer, evaluate, train, read_checkpoint, generate


def smoke(output):
    output = Path(output)
    if output.exists(): raise FileExistsError('choose a new smoke output directory')
    output.mkdir(parents=True)
    records = [dict(paper_id=f'TEST-{i}', group_id=f'test-{i}', version='fixture',
                    license='TEST-ONLY', source_kind='test_fixture',
                    text=('Alpha beta gamma. Conditions matter.\n'*30) + str(i)) for i in range(12)]
    source = output/'fixtures.jsonl'
    source.write_text(''.join(json.dumps(r)+'\n' for r in records))
    prepare(source, output/'corpus', vocab_size=300, split='train', allow_test_data=True, min_chars=1)
    m = ModelConfig(vocab_size=300, dim=32, layers=2, heads=4, kv_heads=2, ffn_dim=64, max_seq_len=128)
    t = TrainConfig(seq_len=32, long_seq_len=None, batch_tokens=64, micro_batch_size=2,
                    warmup_tokens=0, learning_rate=0.003, checkpoint_steps=20)
    cfg = {'model': m.to_dict(), 'training': t.to_dict()}
    corpus = Corpus(output/'corpus'); ds = WindowDataset(corpus, 'train', 32)
    w = Trainer(corpus, m, t, allow_test_data=True)
    before = evaluate(w.model, ds, max_windows=10)['nll']
    train(cfg, output/'corpus', output/'run', steps=8, seconds=120, allow_test_data=True)
    train(cfg, output/'corpus', output/'run', steps=8, seconds=120, resume=True, allow_test_data=True)
    train(cfg, output/'corpus', output/'reference', steps=16, seconds=120, allow_test_data=True)
    state = read_checkpoint(output/'run'); reference = read_checkpoint(output/'reference')
    identical = all(torch.equal(state['model'][n], reference['model'][n]) for n in state['model'])
    w.model.load_state_dict(state['model'])
    after = evaluate(w.model, ds, max_windows=10)['nll']
    # Exercise the actual parent -> supervised correction + replay path as well.
    corrections = [dict(paper_id=f'TEST-correction-{i}', version='fixture', license='TEST-ONLY',
                        source_kind='test_fixture', kind='correction', prompt='What is corrected?',
                        answer='Title: Software test\nAnalysis: Conditions apply.\nLimitations: Test only.',
                        verified=True, evidence=[{'paper_id':'TEST-0', 'version':'fixture', 'span':'fixture'}],
                        verification='This verifies software behavior only, not a scientific finding.') for i in range(2)]
    cpath = output/'corrections.jsonl'; cpath.write_text(''.join(json.dumps(r)+'\n' for r in corrections))
    prepare(cpath, output/'corrections', tokenizer_path=output/'corpus'/'tokenizer.json',
            split='train', allow_test_data=True, min_chars=1)
    update_cfg = {'model': m.to_dict(), 'training': dict(t.to_dict(), replay_ratio=0.2, seq_len=128)}
    update = train(update_cfg, output/'corrections', output/'updated', init_from=output/'run',
                   replay=output/'corpus', steps=4, seconds=120, allow_test_data=True)
    result = dict(test_only=True, loss_before=before, loss_after=after, resume_verified=identical,
                  correction_run=update, scientific_quality='NOT MEASURED',
                  generation=generate(w.model, corpus.tokenizer, 'Alpha', max_new_tokens=8))
    (output/'smoke_report.json').write_text(json.dumps(result, indent=2)+'\n')
    if not identical or not after < before: raise AssertionError('smoke verification failed')
    return result
