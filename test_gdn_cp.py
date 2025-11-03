"""
Minimal test script for GatedDeltaNet Context Parallel (CP) support.
Tests PR: https://github.com/fla-org/flash-linear-attention/pull/691

This script:
1. Initializes GDN parameters (2 layers worth)
2. Runs chunk_gated_delta_rule without CP
3. Runs chunk_gated_delta_rule with CP
4. Compares outputs and gradients for correctness

NOTE: CP is implemented at the ops level (chunk_gated_delta_rule), not the layer level.
The GatedDeltaNet layer class does not yet expose CP - you must call the ops directly.

Usage:
    # Single GPU (no CP, baseline only)
    python test_gdn_cp.py

    # Multi-GPU with CP
    torchrun --nproc_per_node=2 test_gdn_cp.py
    torchrun --nproc_per_node=4 test_gdn_cp.py

Note on GPU compatibility:
    - H100/B100: Can use head_dim=256 (default in fla)
    - A100: Use head_dim=64 or 128 (limited shared memory: 164KB vs 227KB on H100)
    - The default settings below are tuned for A100 compatibility
"""

import os
import argparse

import torch
import torch.distributed as dist


def setup_distributed():
    """Initialize distributed environment if running with torchrun."""
    if "RANK" in os.environ:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])

        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)

        return rank, local_rank, world_size, True
    else:
        return 0, 0, 1, False


def cleanup_distributed():
    """Clean up distributed environment."""
    if dist.is_initialized():
        dist.destroy_process_group()


def print_rank0(msg, rank=0):
    """Print only from rank 0."""
    if rank == 0:
        print(msg)


def create_gdn_inputs(batch_size, seq_len, num_heads, head_dim, device, dtype=torch.bfloat16):
    """
    Create random inputs for chunk_gated_delta_rule.

    When using cu_seqlens (variable-length mode), batch must be 1 with packed sequences.
    So we create shape [1, batch_size * seq_len, H, K] to pack multiple sequences.

    Returns q, k, v, g (gate), beta tensors and cu_seqlens.
    """
    total_len = batch_size * seq_len

    # Use smaller initialization scale to avoid numerical instability
    # The gated delta rule can be sensitive to large values
    scale = 0.1

    # Query, Key: [1, total_len, H, K] - batch=1 with packed sequences
    q = (torch.randn(1, total_len, num_heads, head_dim, device=device, dtype=dtype) * scale).requires_grad_(True)
    k = (torch.randn(1, total_len, num_heads, head_dim, device=device, dtype=dtype) * scale).requires_grad_(True)

    # Value: [1, total_len, H, V]
    v = (torch.randn(1, total_len, num_heads, head_dim, device=device, dtype=dtype) * scale).requires_grad_(True)

    # Gate: [1, total_len, H] - NOTE: gate is per-head, not per-head-dim!
    g = (torch.randn(1, total_len, num_heads, device=device, dtype=dtype) * scale).requires_grad_(True)

    # Beta: [1, total_len, H] - scaling factor, keep in (0, 1) range
    beta = (torch.rand(1, total_len, num_heads, device=device, dtype=dtype) * 0.5 + 0.25).requires_grad_(True)

    # cu_seqlens marks boundaries of each sequence in the packed tensor
    # e.g., for 2 sequences of length 1024: [0, 1024, 2048]
    # Use torch.long as expected by fla
    cu_seqlens = torch.arange(
        0, (batch_size + 1) * seq_len, seq_len,
        dtype=torch.long, device=device
    )

    return q, k, v, g, beta, cu_seqlens


def test_without_cp(q, k, v, g, beta, cu_seqlens, rank=0):
    """Run chunk_gated_delta_rule without context parallel."""
    print_rank0("\n=== Testing WITHOUT Context Parallel ===", rank)

    from fla.ops.gated_delta_rule import chunk_gated_delta_rule

    # Forward
    out, _ = chunk_gated_delta_rule(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        cu_seqlens=cu_seqlens,
    )

    print_rank0(f"Output shape: {out.shape}", rank)

    # Backward
    loss = out.sum()
    loss.backward()

    print_rank0(f"Output mean: {out.mean().item():.6f}", rank)
    print_rank0(f"Q grad mean: {q.grad.mean().item():.6e}", rank)

    return out.detach().clone(), q.grad.detach().clone()


def test_with_cp(q, k, v, g, beta, cu_seqlens, rank, world_size, group):
    """
    Run chunk_gated_delta_rule with context parallel.

    CP splits the packed sequence across ranks. Each rank processes a chunk
    of the total sequence length.
    """
    print_rank0("\n=== Testing WITH Context Parallel ===", rank)

    try:
        from fla.ops.cp import build_cp_context
        from fla.ops.gated_delta_rule import chunk_gated_delta_rule
    except ImportError as e:
        print_rank0(f"CP imports failed: {e}", rank)
        print_rank0("Make sure you have the latest fla with CP support installed.", rank)
        return None, None

    # Split inputs along the packed sequence dimension (dim=1)
    # q, k, v shape: [1, total_len, H, K]
    # g, beta shape: [1, total_len, H]
    total_len = q.shape[1]
    chunk_size = total_len // world_size
    start_idx = rank * chunk_size
    end_idx = (rank + 1) * chunk_size

    q_local = q[:, start_idx:end_idx].clone().detach().requires_grad_(True)
    k_local = k[:, start_idx:end_idx].clone().detach().requires_grad_(True)
    v_local = v[:, start_idx:end_idx].clone().detach().requires_grad_(True)
    g_local = g[:, start_idx:end_idx].clone().detach().requires_grad_(True)  # [1, chunk, H]
    beta_local = beta[:, start_idx:end_idx].clone().detach().requires_grad_(True)  # [1, chunk, H]

    if rank == 0:
        print(f"[Rank {rank}] Local chunk: [{start_idx}:{end_idx}], shape: {q_local.shape}")
    dist.barrier()
    if rank == 1:
        print(f"[Rank {rank}] Local chunk: [{start_idx}:{end_idx}], shape: {q_local.shape}")
    dist.barrier()

    # Build CP context with the global cu_seqlens
    try:
        cp_context = build_cp_context(
            cu_seqlens=cu_seqlens,
            group=group,
        )
        print_rank0(f"CP context built: is_cp_enabled={cp_context.is_cp_enabled}", rank)
        print_rank0(f"CP context cu_seqlens: {cp_context.cu_seqlens}", rank)
    except Exception as e:
        print_rank0(f"Failed to build CP context: {e}", rank)
        import traceback
        traceback.print_exc()
        return None, None

    # Forward with CP
    try:
        out_local, _ = chunk_gated_delta_rule(
            q=q_local,
            k=k_local,
            v=v_local,
            g=g_local,
            beta=beta_local,
            cu_seqlens=cp_context.cu_seqlens,
            cp_context=cp_context,
        )
    except Exception as e:
        print_rank0(f"Forward failed: {e}", rank)
        import traceback
        traceback.print_exc()
        return None, None

    # Backward - each rank computes local gradients
    # Don't all-reduce loss before backward as it breaks autograd
    loss = out_local.sum()
    loss.backward()

    # Gather outputs from all ranks
    gathered_out = [torch.zeros_like(out_local) for _ in range(world_size)]
    dist.all_gather(gathered_out, out_local.contiguous())
    full_out = torch.cat(gathered_out, dim=1)

    # Gather gradients
    gathered_grad = [torch.zeros_like(q_local.grad) for _ in range(world_size)]
    dist.all_gather(gathered_grad, q_local.grad.contiguous())
    full_grad = torch.cat(gathered_grad, dim=1)

    print_rank0(f"Gathered output shape: {full_out.shape}", rank)
    print_rank0(f"Output mean: {full_out.mean().item():.6f}", rank)

    return full_out.detach(), full_grad.detach()


def check_cp_availability():
    """Check what CP functionality is available in the installed fla package."""
    print("\n=== Checking CP Availability ===")

    findings = []

    # Check fla.ops.cp module
    try:
        from fla.ops import cp
        findings.append("✓ fla.ops.cp module exists")

        # Check exports
        exports = getattr(cp, '__all__', dir(cp))
        findings.append(f"  Exports: {[e for e in exports if not e.startswith('_')]}")
    except ImportError as e:
        findings.append(f"✗ fla.ops.cp not available: {e}")
        for f in findings:
            print(f)
        return False

    # Check build_cp_context
    try:
        from fla.ops.cp import build_cp_context
        findings.append("✓ build_cp_context available")
    except ImportError:
        findings.append("✗ build_cp_context not available")

    # Check FLACPContext
    try:
        from fla.ops.cp import FLACPContext
        findings.append("✓ FLACPContext available")
    except ImportError:
        findings.append("✗ FLACPContext not available")

    # Check chunk_gated_delta_rule signature for cp_context
    try:
        from fla.ops.gated_delta_rule import chunk_gated_delta_rule
        import inspect
        sig = inspect.signature(chunk_gated_delta_rule)
        params = list(sig.parameters.keys())
        if 'cp_context' in params:
            findings.append("✓ chunk_gated_delta_rule accepts cp_context parameter")
        else:
            findings.append(f"✗ chunk_gated_delta_rule does NOT have cp_context (params: {params})")
    except Exception as e:
        findings.append(f"✗ Could not inspect chunk_gated_delta_rule: {e}")

    # Check communication functions
    try:
        from fla.ops.cp import comm
        comm_funcs = [n for n in dir(comm) if not n.startswith('_')]
        findings.append(f"✓ Communication functions: {comm_funcs}")
    except ImportError:
        findings.append("✗ fla.ops.cp.comm not available")

    for f in findings:
        print(f)

    return True


def compare_results(out_no_cp, grad_no_cp, out_cp, grad_cp, rank=0, atol=1e-2, rtol=1e-2, grad_atol=5e-2, grad_rtol=5e-2):
    """Compare outputs and gradients between CP and non-CP runs."""
    print_rank0("\n=== Comparing Results ===", rank)

    if out_cp is None:
        print_rank0("CP results not available for comparison.", rank)
        return

    # Compare outputs
    out_diff = (out_no_cp - out_cp).abs()
    out_max_diff = out_diff.max().item()
    out_mean_diff = out_diff.mean().item()

    print_rank0(f"Output max diff: {out_max_diff:.6e}", rank)
    print_rank0(f"Output mean diff: {out_mean_diff:.6e}", rank)

    out_match = torch.allclose(out_no_cp, out_cp, atol=atol, rtol=rtol)
    print_rank0(f"Outputs match (atol={atol}, rtol={rtol}): {out_match}", rank)

    # Compare gradients
    grad_diff = (grad_no_cp - grad_cp).abs()
    grad_max_diff = grad_diff.max().item()
    grad_mean_diff = grad_diff.mean().item()

    print_rank0(f"Gradient max diff: {grad_max_diff:.6e}", rank)
    print_rank0(f"Gradient mean diff: {grad_mean_diff:.6e}", rank)

    grad_match = torch.allclose(grad_no_cp, grad_cp, atol=grad_atol, rtol=grad_rtol)
    print_rank0(f"Gradients match (atol={grad_atol}, rtol={grad_rtol}): {grad_match}", rank)

    if out_match and grad_match:
        print_rank0("\n✓ CP implementation is correct!", rank)
    else:
        print_rank0("\n✗ Mismatch detected between CP and non-CP results.", rank)


def main():
    parser = argparse.ArgumentParser(description="Test GDN Context Parallel")
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=64,
                        help="Head dimension. Use 64/128 for A100, 256 for H100/B100")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--check-only", action="store_true",
                        help="Only check CP availability, don't run tests")
    args = parser.parse_args()

    # Setup
    rank, local_rank, world_size, is_distributed = setup_distributed()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    print_rank0(f"Running on {world_size} GPU(s), distributed={is_distributed}", rank)
    print_rank0(f"Config: num_heads={args.num_heads}, head_dim={args.head_dim}", rank)
    print_rank0(f"Input: batch_size={args.batch_size}, seq_len={args.seq_len}", rank)

    # Check CP availability
    if rank == 0:
        cp_available = check_cp_availability()
        if not cp_available:
            print("\nCP not available. Install latest fla:")
            print("pip uninstall fla-core flash-linear-attention -y")
            print("pip install -U git+https://github.com/fla-org/flash-linear-attention")

    if args.check_only:
        cleanup_distributed()
        return

    # Set seed for reproducibility
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # Ensure seq_len is divisible by world_size for CP
    assert args.seq_len % world_size == 0, \
        f"seq_len ({args.seq_len}) must be divisible by world_size ({world_size})"

    # Create inputs (returns packed sequences with batch=1)
    print_rank0("\n=== Creating Inputs ===", rank)
    total_seq_len = args.batch_size * args.seq_len
    print_rank0(f"Packed sequence: {args.batch_size} sequences x {args.seq_len} tokens = {total_seq_len} total", rank)

    q, k, v, g, beta, cu_seqlens = create_gdn_inputs(
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        num_heads=args.num_heads,
        head_dim=args.head_dim,
        device=device,
    )

    # Broadcast inputs to all ranks for consistency
    if is_distributed:
        for tensor in [q, k, v, g, beta]:
            dist.broadcast(tensor.data, src=0)

    print_rank0(f"Input shape: {q.shape} (batch=1, packed_len={total_seq_len}, heads={args.num_heads}, head_dim={args.head_dim})", rank)
    print_rank0(f"cu_seqlens: {cu_seqlens}", rank)

    # Test without CP
    out_no_cp, grad_no_cp = test_without_cp(q, k, v, g, beta, cu_seqlens, rank)

    # Test with CP (only if distributed)
    if is_distributed and world_size > 1:
        group = dist.group.WORLD

        # Need fresh tensors for CP test
        q2, k2, v2, g2, beta2, cu_seqlens2 = create_gdn_inputs(
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            num_heads=args.num_heads,
            head_dim=args.head_dim,
            device=device,
        )
        # Copy data from original
        with torch.no_grad():
            q2.copy_(q)
            k2.copy_(k)
            v2.copy_(v)
            g2.copy_(g)
            beta2.copy_(beta)

        out_cp, grad_cp = test_with_cp(q2, k2, v2, g2, beta2, cu_seqlens, rank, world_size, group)

        # Compare results on rank 0
        if rank == 0:
            compare_results(out_no_cp, grad_no_cp, out_cp, grad_cp, rank)
    else:
        print_rank0("\nSkipping CP test (requires distributed run with world_size > 1)", rank)
        print_rank0("Run with: torchrun --nproc_per_node=2 test_gdn_cp.py", rank)

    # Cleanup
    cleanup_distributed()
    print_rank0("\n=== Test Complete ===", rank)


if __name__ == "__main__":
    main()
