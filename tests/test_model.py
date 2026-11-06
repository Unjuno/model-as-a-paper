import pytest
import torch


def api():
    try:
        from model_as_a_paper.model import LanguageModel, ModelConfig
    except ImportError:
        pytest.fail('MAP model implementation is missing')
    return LanguageModel, ModelConfig


def small():
    Model, Config = api()
    torch.manual_seed(7)
    return Model(Config(vocab_size=260, dim=32, layers=2, heads=4, kv_heads=2, ffn_dim=64, max_seq_len=32))


def test_exact_parameter_count_and_tied_identity():
    Model, Config = api()
    with torch.device('meta'):
        m = Model(Config())
    assert sum(p.numel() for p in m.parameters()) == 44_577_280
    assert m.lm_head.weight is m.embedding.weight
    assert len({id(p) for p in m.parameters()}) == len(list(m.parameters()))


def test_causal_no_future_leak():
    m = small().eval()
    x = torch.randint(4, 260, (2, 12))
    y = x.clone(); y[:, 7:] = torch.randint(4, 260, (2, 5))
    with torch.no_grad():
        a, _ = m(x); b, _ = m(y)
    torch.testing.assert_close(a[:, :7], b[:, :7], rtol=0, atol=0)


def test_incremental_cache_matches_full_and_multitoken_cache():
    m = small().eval()
    x = torch.randint(4, 260, (1, 12))
    with torch.no_grad():
        full, _ = m(x)
        _, cache = m(x[:, :5], use_cache=True)
        last, _ = m(x[:, 5:], cache=cache, use_cache=True)
    torch.testing.assert_close(last, full[:, 5:], atol=2e-6, rtol=1e-5)


def test_gqa_matches_explicit_kv_repeat():
    m = small().eval()
    x = torch.randint(4, 260, (2, 7))
    with torch.no_grad():
        a, _ = m(x)
        for b in m.blocks:
            b.attention.reference_gqa = True
        ref, _ = m(x)
    torch.testing.assert_close(a, ref, atol=2e-6, rtol=1e-5)


def test_gradients_finite_and_all_parameters_used():
    m = small()
    x = torch.randint(4, 260, (2, 8))
    logits, _ = m(x)
    loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), x.flatten())
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())


def test_invalid_shape_and_context_rejected():
    _, Config = api()
    with pytest.raises(ValueError):
        Config(heads=8, kv_heads=3)
    with pytest.raises(ValueError):
        small()(torch.ones(1, 33, dtype=torch.long))
