import pytest
import torch
import torch.nn.functional as F
from model_as_a_paper.model import ModelConfig, PaperLM


def tiny():
    return ModelConfig(vocab_size=300, dim=32, layers=2, heads=4, kv_heads=2,
                       ffn_dim=64, max_seq_len=64)


def test_full_model_parameter_count_and_unique_embedding():
    with torch.device('meta'):
        m = PaperLM(ModelConfig())
    assert sum(p.numel() for p in m.parameters()) == 44_577_280
    assert len(list(m.parameters())) == len({id(p) for p in m.parameters()})
    assert m.parameter_count() == 44_577_280


def test_causality_future_changes_do_not_change_prefix():
    torch.manual_seed(2)
    m = PaperLM(tiny()).eval()
    x = torch.randint(4, 300, (2, 12))
    y = x.clone(); y[:, 6:] = torch.randint(4, 300, (2, 6))
    assert torch.allclose(m(x)[0][:, :6], m(y)[0][:, :6], atol=1e-6)


def test_cache_matches_full_forward_including_multi_token_append():
    torch.manual_seed(3)
    m = PaperLM(tiny()).eval()
    x = torch.randint(4, 300, (1, 12))
    with torch.no_grad():
        full, _ = m(x)
        a, cache = m(x[:, :5], use_cache=True)
        b, cache = m(x[:, 5:8], cache=cache, use_cache=True)
        pieces = [a, b]
        for i in range(8, 12):
            p, cache = m(x[:, i:i+1], cache=cache, use_cache=True); pieces.append(p)
    assert torch.allclose(full, torch.cat(pieces, dim=1), atol=2e-6)
    assert cache[0][0].shape == (1, 2, 12, 8)


def test_gqa_matches_explicit_repeat_reference():
    m = PaperLM(tiny()).eval()
    att = m.blocks[0].attn
    x = torch.randn(2, 7, 32)
    q, k, v = att.project(x, 0)
    k = k.repeat_interleave(2, dim=1); v = v.repeat_interleave(2, dim=1)
    ref = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    ref = att.out(ref.transpose(1, 2).reshape(2, 7, 32))
    got, _ = att(x)
    assert torch.allclose(got, ref, atol=1e-6)


def test_all_trainable_parameters_receive_finite_gradients():
    m = PaperLM(tiny())
    x = torch.randint(4, 300, (2, 9))
    logits, _ = m(x[:, :-1])
    F.cross_entropy(logits.reshape(-1, 300), x[:, 1:].reshape(-1)).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())


@pytest.mark.parametrize('kwargs', [{'heads':3}, {'kv_heads':3}, {'dim':0},
                                    {'layers':0}, {'ffn_dim':0}, {'max_seq_len':0}])
def test_invalid_shapes_fail_before_training(kwargs):
    with pytest.raises(ValueError):
        ModelConfig(**kwargs)


def test_reject_cache_during_training_and_context_overflow():
    m = PaperLM(tiny())
    with pytest.raises(ValueError): m(torch.ones(1, 2, dtype=torch.long), use_cache=True)
    with pytest.raises(ValueError): m(torch.ones(1, 65, dtype=torch.long))
