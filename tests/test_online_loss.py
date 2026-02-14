"""
Correctness tests for fms_fsdp/fsdp2/online_loss.py.

Three layers of coverage:

  1. Non-parallel numerics (single process, no DTensor):
       pytest tests/test_online_loss.py -k "not parallel"

  2. Gradient correctness via gradcheck (single process, float32):
       pytest tests/test_online_loss.py -k "gradcheck"

  3. Vocab-parallel correctness (spawns 2 CPU processes with gloo):
       pytest tests/test_online_loss.py -k "parallel"
"""

import os
import torch
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
import pytest

from fms_fsdp.fsdp2.online_loss import (
    _stream_logsumexp_from_hidden,
    streaming_ce_and_zloss,
    streaming_forward_kl,
)

# ── small but non-trivial dimensions ──────────────────────────────────────────
B, T, D, V = 2, 8, 32, 128


# ── reference implementations ─────────────────────────────────────────────────

def _ref_logZ(h, W):
    """Exact logsumexp via PyTorch."""
    B, T, d = h.shape
    return (h.reshape(-1, d) @ W.T).logsumexp(-1).reshape(B, T)


def _ref_ce(h, W, labels, ignore_index=-100):
    """Exact CE via PyTorch cross_entropy."""
    B, T, d = h.shape
    logits = h.reshape(-1, d) @ W.T    # (BT, V)
    return F.cross_entropy(logits, labels.reshape(-1), ignore_index=ignore_index)


def _ref_kl(h_t, W_t, h_s, W_s, mask_bt, temperature):
    """Exact forward KL: KL(pt || ps) with temperature, via full materialisation."""
    B, T, d = h_s.shape
    with torch.no_grad():
        logits_t = (h_t.reshape(-1, d) @ W_t.T / temperature).reshape(B, T, -1)
        logits_s = (h_s.reshape(-1, d) @ W_s.T / temperature).reshape(B, T, -1)
    log_pt = logits_t - logits_t.logsumexp(-1, keepdim=True)
    log_ps = logits_s - logits_s.logsumexp(-1, keepdim=True)
    pt = log_pt.exp()
    kl_pointwise = (pt * (log_pt - log_ps)).sum(-1)    # (B, T)
    denom = mask_bt.sum().clamp_min(1)
    return (kl_pointwise * mask_bt).sum() / denom * (temperature ** 2)


# ══════════════════════════════════════════════════════════════════════════════
# 1.  Non-parallel numeric tests
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("vchunk", [16, 64, V])
def test_logsumexp_matches_reference(vchunk):
    torch.manual_seed(0)
    h = torch.randn(B, T, D)
    W = torch.randn(V, D)

    got = _stream_logsumexp_from_hidden(h, W, vchunk=vchunk, pg=None)
    ref = _ref_logZ(h, W)

    assert torch.allclose(got, ref, atol=1e-5), \
        f"logsumexp mismatch (vchunk={vchunk}): max err={(got - ref).abs().max():.2e}"


@pytest.mark.parametrize("vchunk", [16, 64, V])
def test_ce_loss_matches_reference(vchunk):
    torch.manual_seed(1)
    h = torch.randn(B, T, D)
    W = torch.randn(V, D)
    labels = torch.randint(0, V, (B, T))
    labels[0, 3] = -100    # one ignored position

    loss, _, _, _ = streaming_ce_and_zloss(
        h, W, labels, ignore_index=-100, zl_coeff=0.0, vchunk=vchunk
    )
    ref = _ref_ce(h, W, labels)

    assert torch.allclose(loss, ref, atol=1e-5), \
        f"CE mismatch (vchunk={vchunk}): got {loss.item():.6f}, ref {ref.item():.6f}"


def test_zloss_magnitude():
    torch.manual_seed(2)
    h = torch.randn(B, T, D)
    W = torch.randn(V, D)
    labels = torch.randint(0, V, (B, T))

    loss0, _, zl0, _ = streaming_ce_and_zloss(h, W, labels, -100, zl_coeff=0.0, vchunk=32)
    loss1, _, zl1, _ = streaming_ce_and_zloss(h, W, labels, -100, zl_coeff=1e-3, vchunk=32)

    assert zl0.item() == 0.0,  "zloss should be 0 when coeff=0"
    assert zl1.item() > 0.0,   "zloss should be >0 when coeff>0"
    assert not torch.allclose(loss0, loss1), "total loss should differ with zloss"


@pytest.mark.parametrize("temperature", [1.0, 2.0])
def test_kl_matches_reference(temperature):
    torch.manual_seed(3)
    h_t = torch.randn(B, T, D)
    W_t = torch.randn(V, D)
    h_s = torch.randn(B, T, D, requires_grad=True)
    W_s = torch.randn(V, D, requires_grad=True)
    mask = torch.ones(B, T, dtype=torch.bool)
    mask[0, -1] = False    # one masked position

    kl = streaming_forward_kl(h_t, W_t, h_s, W_s, mask, temperature=temperature, vchunk=32)
    ref = _ref_kl(h_t, W_t, h_s.detach(), W_s.detach(), mask, temperature)

    assert torch.allclose(kl, ref, atol=1e-4), \
        f"KL mismatch (T={temperature}): got {kl.item():.6f}, ref {ref.item():.6f}"


# ══════════════════════════════════════════════════════════════════════════════
# 2.  Gradient correctness (fp32) — compare to dense PyTorch reference
# ══════════════════════════════════════════════════════════════════════════════

def _dense_ce_and_zloss_ref(h, W, labels, ignore_index=-100, zl_coeff=0.0):
    B, T, d = h.shape
    logits = h.reshape(-1, d) @ W.T                       # (BT, V)

    # CE: mean over non-ignored tokens (PyTorch matches your ce_val.mean())
    ce = F.cross_entropy(logits, labels.reshape(-1), ignore_index=ignore_index)

    if zl_coeff == 0.0:
        return ce

    # zloss: zl_coeff * mean(logZ^2) over ALL (B*T) positions (no mask)
    logZ = logits.logsumexp(-1).reshape(B, T)             # (B, T)
    zl = zl_coeff * (logZ.pow(2).mean())

    return ce + zl


def _dense_forward_kl_ref(h_t, W_t, h_s, W_s, mask_bt, temperature):
    """
    Differentiable dense reference for:
      KL(p_t || p_s) with temperature, reduced over mask and scaled by T^2.
    Teacher can be treated as constant; student remains differentiable.
    """
    B, T, d = h_s.shape

    # teacher logits: constant
    with torch.no_grad():
        logits_t = (h_t.reshape(-1, d) @ W_t.T / temperature).reshape(B, T, -1)
        log_pt = logits_t - logits_t.logsumexp(-1, keepdim=True)
        pt = log_pt.exp()

    # student logits: differentiable
    logits_s = (h_s.reshape(-1, d) @ W_s.T / temperature).reshape(B, T, -1)
    log_ps = logits_s - logits_s.logsumexp(-1, keepdim=True)

    kl_pointwise = (pt * (log_pt - log_ps)).sum(-1)       # (B, T)
    denom = mask_bt.sum().clamp_min(1)
    return (kl_pointwise * mask_bt).sum() / denom * (temperature ** 2)


@pytest.mark.parametrize("zl_coeff", [0.0, 1e-2])
def test_fp32_grads_ce_match_dense_reference(zl_coeff):
    torch.manual_seed(4 if zl_coeff == 0.0 else 5)
    h = torch.randn(2, 4, 16, dtype=torch.float32, requires_grad=True)
    W = torch.randn(32, 16, dtype=torch.float32, requires_grad=True)
    labels = torch.randint(0, 32, (2, 4))
    labels[0, 0] = -100

    # streaming loss + grads
    loss_s, _, _, _ = streaming_ce_and_zloss(
        h, W, labels, ignore_index=-100, zl_coeff=zl_coeff, vchunk=16
    )
    gh_s, gW_s = torch.autograd.grad(loss_s, (h, W), retain_graph=False)

    # dense reference loss + grads
    loss_r = _dense_ce_and_zloss_ref(h, W, labels, ignore_index=-100, zl_coeff=zl_coeff)
    gh_r, gW_r = torch.autograd.grad(loss_r, (h, W), retain_graph=False)

    # fp32 tolerances: a bit looser than fp64, but still tight
    assert torch.allclose(gh_s, gh_r, atol=2e-5, rtol=2e-4), \
        f"grad_h max err={(gh_s - gh_r).abs().max():.2e}"
    assert torch.allclose(gW_s, gW_r, atol=2e-5, rtol=2e-4), \
        f"grad_W max err={(gW_s - gW_r).abs().max():.2e}"


def test_fp32_grads_kl_match_dense_reference():
    torch.manual_seed(6)
    h_t = torch.randn(2, 4, 16, dtype=torch.float32)
    W_t = torch.randn(32, 16, dtype=torch.float32)

    h_s = torch.randn(2, 4, 16, dtype=torch.float32, requires_grad=True)
    W_s = torch.randn(32, 16, dtype=torch.float32, requires_grad=True)

    mask = torch.ones(2, 4, dtype=torch.bool)

    # streaming
    kl_s = streaming_forward_kl(h_t, W_t, h_s, W_s, mask, temperature=1.0, vchunk=16)
    gh_s, gW_s = torch.autograd.grad(kl_s, (h_s, W_s), retain_graph=False)

    # dense ref
    kl_r = _dense_forward_kl_ref(h_t, W_t, h_s, W_s, mask, temperature=1.0)
    gh_r, gW_r = torch.autograd.grad(kl_r, (h_s, W_s), retain_graph=False)

    assert torch.allclose(gh_s, gh_r, atol=2e-5, rtol=2e-4), \
        f"KL grad_h max err={(gh_s - gh_r).abs().max():.2e}"
    assert torch.allclose(gW_s, gW_r, atol=2e-5, rtol=2e-4), \
        f"KL grad_W max err={(gW_s - gW_r).abs().max():.2e}"


# ══════════════════════════════════════════════════════════════════════════════
# 3.  Vocab-parallel correctness (2 CPU processes)
# ══════════════════════════════════════════════════════════════════════════════

def _vp_worker_ce(rank, world_size, results):
    """
    Rank r holds vocab rows [r*V_local : (r+1)*V_local].
    The vocab-parallel CE loss must match the single-rank reference.
    grad_h from the VP path must match the single-rank grad_h.
    """
    os.environ.update({"MASTER_ADDR": "127.0.0.1", "MASTER_PORT": "29502"})
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.tensor import DTensor, Shard

    torch.manual_seed(42)    # same seed on all ranks → identical h / W / labels
    h_ref  = torch.randn(B, T, D, requires_grad=True)
    W_full = torch.randn(V, D)
    labels = torch.randint(0, V, (B, T))
    labels[0, 0] = -100

    # ── single-rank reference ──────────────────────────────────────────────
    W_ref = W_full.clone().requires_grad_(True)
    loss_ref, _, _, _ = streaming_ce_and_zloss(
        h_ref, W_ref, labels, ignore_index=-100, zl_coeff=0.0, vchunk=16
    )
    loss_ref.backward()
    grad_h_ref = h_ref.grad.clone()
    # Take this rank's slice of the reference W gradient for later comparison.
    V_local = V // world_size
    v0, v1 = rank * V_local, (rank + 1) * V_local
    grad_W_ref_local = W_ref.grad[v0:v1].clone()

    # ── vocab-parallel path ───────────────────────────────────────────────
    mesh    = init_device_mesh("cpu", (world_size,))
    W_local = W_full[v0:v1].clone()
    W_dt    = DTensor.from_local(W_local, mesh, [Shard(0)]).requires_grad_(True)

    h_vp = h_ref.detach().requires_grad_(True)
    loss_vp, _, _, _ = streaming_ce_and_zloss(
        h_vp, W_dt, labels, ignore_index=-100, zl_coeff=0.0, vchunk=16
    )

    # Loss value must match.
    assert torch.allclose(loss_vp, loss_ref.detach(), atol=1e-5), \
        f"rank {rank}: loss {loss_vp.item():.6f} vs ref {loss_ref.item():.6f}"

    loss_vp.backward()

    # grad_h must match the single-rank result.
    assert torch.allclose(h_vp.grad, grad_h_ref, atol=1e-5), \
        f"rank {rank}: grad_h max err={(h_vp.grad - grad_h_ref).abs().max():.2e}"

    # grad_W for this rank's shard must match the corresponding slice.
    grad_W_local = W_dt.grad.to_local()
    assert torch.allclose(grad_W_local, grad_W_ref_local, atol=1e-5), \
        f"rank {rank}: grad_W local shard max err={(grad_W_local - grad_W_ref_local).abs().max():.2e}"

    results[rank] = "OK"
    dist.destroy_process_group()


def _vp_worker_kl(rank, world_size, results):
    """
    Vocab-parallel forward KL must match the single-rank reference.
    """
    os.environ.update({"MASTER_ADDR": "127.0.0.1", "MASTER_PORT": "29503"})
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.tensor import DTensor, Shard

    torch.manual_seed(7)
    h_t    = torch.randn(B, T, D)
    W_t    = torch.randn(V, D)
    h_s    = torch.randn(B, T, D, requires_grad=True)
    W_s    = torch.randn(V, D)
    mask   = torch.ones(B, T, dtype=torch.bool)
    mask[0, -1] = False

    # ── single-rank reference ──────────────────────────────────────────────
    W_s_ref = W_s.clone().requires_grad_(True)
    kl_ref  = streaming_forward_kl(h_t, W_t, h_s, W_s_ref, mask, temperature=1.0, vchunk=16)
    kl_ref.backward()
    grad_h_ref = h_s.grad.clone()

    V_local = V // world_size
    v0, v1  = rank * V_local, (rank + 1) * V_local

    # ── vocab-parallel path ───────────────────────────────────────────────
    mesh   = init_device_mesh("cpu", (world_size,))
    W_t_dt = DTensor.from_local(W_t[v0:v1].clone(), mesh, [Shard(0)])
    W_s_dt = DTensor.from_local(W_s[v0:v1].clone(), mesh, [Shard(0)]).requires_grad_(True)

    h_s_vp = h_s.detach().requires_grad_(True)
    kl_vp  = streaming_forward_kl(h_t, W_t_dt, h_s_vp, W_s_dt, mask, temperature=1.0, vchunk=16)

    assert torch.allclose(kl_vp, kl_ref.detach(), atol=1e-4), \
        f"rank {rank}: KL {kl_vp.item():.6f} vs ref {kl_ref.item():.6f}"

    kl_vp.backward()

    assert torch.allclose(h_s_vp.grad, grad_h_ref, atol=1e-5), \
        f"rank {rank}: KL grad_h max err={(h_s_vp.grad - grad_h_ref).abs().max():.2e}"

    results[rank] = "OK"
    dist.destroy_process_group()


def _run_parallel(worker_fn, world_size=2):
    ctx     = mp.get_context("spawn")
    manager = ctx.Manager()
    results = manager.dict()
    procs   = [ctx.Process(target=worker_fn, args=(r, world_size, results))
               for r in range(world_size)]
    for p in procs:
        p.start()
    for p in procs:
        p.join()
        assert p.exitcode == 0, f"worker process {p.pid} exited with code {p.exitcode}"
    assert all(results.get(r) == "OK" for r in range(world_size)), \
        f"some workers did not finish cleanly: {dict(results)}"


def test_vocab_parallel_ce():
    _run_parallel(_vp_worker_ce, world_size=2)


def test_vocab_parallel_kl():
    _run_parallel(_vp_worker_kl, world_size=2)
