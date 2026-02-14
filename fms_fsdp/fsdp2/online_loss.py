import torch
import torch.nn.functional as F
import torch.distributed._functional_collectives as funcol

# Keep the online Cross-Entropy loss, Z-loss and KL-divergence


def _get_full_weight(W):
    """
    If W is an FSDP2 DTensor sharded along dim 0 (vocab dimension):
      - All-gathers to the full (V, d) tensor on every rank.
      - Returns (W_full, pg, mesh, placements) where pg / mesh / placements are
        saved for the reduce-scatter in the backward pass.

    If W is a plain tensor, returns (W, None, None, None).

    Why all-gather instead of vocab-parallel all-reduce
    ----------------------------------------------------
    When context parallelism (CP) is active, each CP rank holds a *different*
    sequence chunk.  A vocab-parallel approach (each rank keeps W_local and
    all-reduces partial logits) requires all ranks to share the same hidden
    state h_s, which is false under CP.  All-gathering W so every rank has the
    full vocabulary lets each rank compute a *locally correct* softmax for its
    own sequence chunk with no cross-rank communication during the CE/KL
    computation itself.
    """
    try:
        from torch.distributed.tensor import DTensor, Shard
    except ImportError:
        return W, None, None, None

    if not isinstance(W, DTensor):
        return W, None, None, None

    mesh = W.device_mesh
    placements = W.placements

    shard_mesh_dim = None
    for mesh_dim, p in enumerate(placements):
        if isinstance(p, Shard) and p.dim == 0:
            shard_mesh_dim = mesh_dim
            break

    if shard_mesh_dim is None:
        return W.to_local(), None, None, None

    W_local = W.to_local()                                # (V_local, d)
    pg = mesh.get_group(mesh_dim=shard_mesh_dim)

    W_full = funcol.all_gather_tensor(W_local, gather_dim=0, group=pg)
    return W_full, pg, mesh, placements


def _stream_logsumexp(h, W, vchunk: int):
    """
    Streaming logsumexp: logZ = log(Σ_v exp(h @ w_v^T)), shape (B, T) fp32.
    W is the full (V, d) weight on this rank — no cross-rank communication.
    """
    V = W.shape[0]
    h32 = h.float()
    W32 = W.float()

    m = None
    for v0 in range(0, V, vchunk):
        v1 = min(v0 + vchunk, V)
        z_max = (h32 @ W32[v0:v1].T).amax(dim=-1)
        m = z_max if m is None else torch.maximum(m, z_max)

    s = None
    for v0 in range(0, V, vchunk):
        v1 = min(v0 + vchunk, V)
        z_sum = (h32 @ W32[v0:v1].T - m.unsqueeze(-1)).exp_().sum(dim=-1)
        s = z_sum if s is None else (s + z_sum)

    return m + s.log()


class _StreamingCEZLossFunc(torch.autograd.Function):
    """
    Custom autograd for streaming CE + Z-loss so that peak memory
    is O(B*T*vchunk) instead of O(B*T*V).

    When W_s is an FSDP2 Shard(0) DTensor:
      Forward  : all_gather(W_local) → W_full once; each rank computes CE for
                 its own (possibly different) sequence chunk independently.
      Backward : compute grad_W_full locally, then reduce_scatter → grad_W_local;
                 return DTensor.from_local(grad_W_local) so FSDP2 receives a
                 properly-sharded gradient without double-reducing.
                 grad_h needs no cross-rank communication.

    This is correct under context parallelism (CP) because each rank always
    has the full vocabulary and can compute an exact softmax for its tokens.
    """

    @staticmethod
    def forward(ctx, h_s, W_s, labels, ignore_index, zl_coeff, vchunk, _diagnostics):
        B, T, d = h_s.shape
        BT = B * T
        zl_coeff = float(zl_coeff)

        W, pg, W_s_mesh, W_s_placements = _get_full_weight(W_s)

        logZ = _stream_logsumexp(h_s, W, vchunk)           # (B, T) fp32

        labels_flat = labels.reshape(-1)
        mask = labels_flat.ne(ignore_index)
        N = mask.sum().clamp_min(1)

        h_flat = h_s.reshape(-1, d)
        h_sel = h_flat[mask]
        y_sel = labels_flat[mask].long()
        logZ_sel = logZ.reshape(-1)[mask]

        W32 = W.float()
        zy = (h_sel.float() * W32[y_sel]).sum(dim=-1)

        ce_val = (-(zy - logZ_sel)).mean()

        zloss_val = torch.zeros((), device=h_s.device, dtype=torch.float32)
        if zl_coeff != 0.0:
            zloss_val = zl_coeff * logZ.pow(2).mean()

        loss = ce_val + zloss_val

        if _diagnostics is not None:
            _diagnostics['ce'] = ce_val.detach()
            _diagnostics['zloss'] = zloss_val.detach()
            _diagnostics['logZ'] = logZ

        ctx.save_for_backward(h_s, W, logZ, labels)
        ctx.ignore_index = ignore_index
        ctx.zl_coeff = zl_coeff
        ctx.vchunk = vchunk
        ctx.N = int(N.item())
        ctx.BT = BT
        ctx.pg = pg
        ctx.W_s_mesh = W_s_mesh
        ctx.W_s_placements = W_s_placements
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        h_s, W, logZ, labels = ctx.saved_tensors
        ignore_index = ctx.ignore_index
        zl_coeff = ctx.zl_coeff
        vchunk = ctx.vchunk
        N = ctx.N
        BT = ctx.BT
        V = W.shape[0]
        d = h_s.shape[2]
        g = float(grad_output)
        pg = ctx.pg

        labels_flat = labels.reshape(-1)
        mask = labels_flat.ne(ignore_index)
        logZ_flat = logZ.reshape(-1)
        y_all = labels_flat.long()

        coeff_p = mask.float() * (g / N)
        if zl_coeff != 0.0:
            coeff_p = coeff_p + (2.0 * zl_coeff / BT) * logZ_flat * g

        sparse_pos = mask.nonzero(as_tuple=True)[0]
        sparse_v = y_all[sparse_pos]                        # indices into full V

        h_flat = h_s.reshape(-1, d)
        h_flat32 = h_flat.float()
        W32 = W.float()

        grad_h_flat32 = torch.zeros(h_flat.shape, device=h_s.device, dtype=torch.float32)
        grad_W32 = torch.zeros(W.shape, device=h_s.device, dtype=torch.float32)

        for v0 in range(0, V, vchunk):
            v1 = min(v0 + vchunk, V)

            ps = (h_flat32 @ W32[v0:v1].T).sub_(logZ_flat.unsqueeze(-1)).exp_()
            delta = ps.mul_(coeff_p.unsqueeze(-1))

            in_chunk = (sparse_v >= v0) & (sparse_v < v1)
            if in_chunk.any():
                sel_pos = sparse_pos[in_chunk]
                sel_v = sparse_v[in_chunk] - v0
                delta[sel_pos, sel_v] -= g / N

            grad_h_flat32.add_(delta @ W32[v0:v1])
            grad_W32[v0:v1].add_(delta.T @ h_flat32)

        # grad_h: each rank has the correct gradient for its own sequence chunk;
        # no cross-rank reduction needed.
        grad_h_out = grad_h_flat32.to(h_s.dtype).reshape_as(h_s)

        if pg is not None:
            # reduce_scatter: sum each rank's local grad_W contribution (from its
            # own sequence tokens) and scatter so rank i gets grad_W for its vocab
            # shard — the dual of the forward all-gather.
            grad_W_local = funcol.reduce_scatter_tensor(
                grad_W32, reduceOp="sum", scatter_dim=0, group=pg
            ).to(W.dtype)
            from torch.distributed.tensor import DTensor
            grad_W_out = DTensor.from_local(grad_W_local, ctx.W_s_mesh, ctx.W_s_placements)
        else:
            grad_W_out = grad_W32.to(W.dtype)

        return grad_h_out, grad_W_out, None, None, None, None, None


def streaming_ce_and_zloss(h_s, W_s, labels, ignore_index: int, zl_coeff: float, vchunk: int):
    """
    Exact CE + zloss without materializing logits.
    - CE: mean over non-ignored tokens
    - zloss: zl_coeff * mean(logZ^2) over all positions
    Returns: (loss_total, ce_loss_scalar, zloss_scalar, logZ_s)

    Uses a custom autograd.Function so peak memory is O(B*T*vchunk).
    When W_s is an FSDP2 Shard(0) DTensor, all-gathers to full vocab before
    computing and reduce-scatters the weight gradient in backward.
    """
    _diagnostics = {}
    loss = _StreamingCEZLossFunc.apply(
        h_s, W_s, labels, ignore_index, zl_coeff, vchunk, _diagnostics,
    )
    return loss, _diagnostics['ce'], _diagnostics['zloss'], _diagnostics['logZ']


class _StreamingForwardKLFunc(torch.autograd.Function):
    """
    Custom autograd for streaming forward KL so that peak memory
    is O(B*T*vchunk) instead of O(B*T*V).

    When W_s / W_t are FSDP2 Shard(0) DTensors, all-gathers both weights to
    the full vocabulary on each rank.  Each rank computes KL(pt||ps) for its
    own sequence chunk independently — correct under CP.  In backward, a
    reduce_scatter on grad_W_s returns the sharded gradient to FSDP2.
    """

    @staticmethod
    def forward(ctx, h_s, W_s, h_t, W_t, mask_bt, temperature, vchunk):
        Ttemp = float(temperature)

        W_s_full, pg, W_s_mesh, W_s_placements = _get_full_weight(W_s)
        W_t_full, _, _, _                       = _get_full_weight(W_t)

        h_s_sc = h_s / Ttemp
        h_t_sc = h_t / Ttemp
        h_t32 = h_t_sc.float()
        h_s32 = h_s_sc.float()
        W_t32 = W_t_full.float()
        W_s32 = W_s_full.float()

        logZ_t = _stream_logsumexp(h_t_sc, W_t_full, vchunk)
        logZ_s = _stream_logsumexp(h_s_sc, W_s_full, vchunk)

        V = W_s_full.shape[0]
        denom = mask_bt.sum().clamp_min(1)

        kl_sum = torch.zeros((), device=h_s.device, dtype=torch.float32)
        for v0 in range(0, V, vchunk):
            v1 = min(v0 + vchunk, V)
            logpt = h_t32 @ W_t32[v0:v1].T - logZ_t.unsqueeze(-1)
            pt = logpt.exp()
            logps = h_s32 @ W_s32[v0:v1].T - logZ_s.unsqueeze(-1)
            kl_sum += ((pt * (logpt - logps)).sum(-1) * mask_bt).sum()

        kl = (kl_sum / denom) * (Ttemp * Ttemp)

        ctx.save_for_backward(h_s, W_s_full, h_t, W_t_full, mask_bt, logZ_s, logZ_t)
        ctx.temperature = Ttemp
        ctx.vchunk = vchunk
        ctx.pg = pg
        ctx.W_s_mesh = W_s_mesh
        ctx.W_s_placements = W_s_placements
        return kl

    @staticmethod
    def backward(ctx, grad_output):
        h_s, W_s_full, h_t, W_t_full, mask_bt, logZ_s, logZ_t = ctx.saved_tensors
        Ttemp = ctx.temperature
        vchunk = ctx.vchunk
        pg = ctx.pg
        d = h_s.shape[2]
        V = W_s_full.shape[0]

        denom = mask_bt.sum().clamp_min(1)
        scale = grad_output * (Ttemp * Ttemp) / denom

        h_s_sc = h_s / Ttemp
        h_t_sc = h_t / Ttemp
        h_t32 = h_t_sc.float()
        h_s32 = h_s_sc.float()
        W_t32 = W_t_full.float()
        W_s32 = W_s_full.float()

        grad_h_s32 = torch.zeros(h_s.shape, device=h_s.device, dtype=torch.float32)
        grad_W_s32 = torch.zeros(W_s_full.shape, device=h_s.device, dtype=torch.float32)

        mask_f = mask_bt.unsqueeze(-1).float()

        for v0 in range(0, V, vchunk):
            v1 = min(v0 + vchunk, V)
            vc = v1 - v0

            pt = (h_t32 @ W_t32[v0:v1].T).sub_(logZ_t.unsqueeze(-1)).exp_()
            ps = (h_s32 @ W_s32[v0:v1].T).sub_(logZ_s.unsqueeze(-1)).exp_()

            diff = (ps - pt).mul_(mask_f).mul_(scale)       # (B, T, vc)

            grad_h_s32.add_((diff @ W_s32[v0:v1]) / Ttemp)
            grad_W_s32[v0:v1].add_(diff.reshape(-1, vc).T @ h_s32.reshape(-1, d))

        grad_h_out = grad_h_s32.to(h_s.dtype)

        if pg is not None:
            grad_W_local = funcol.reduce_scatter_tensor(
                grad_W_s32, reduceOp="sum", scatter_dim=0, group=pg
            ).to(W_s_full.dtype)
            from torch.distributed.tensor import DTensor
            grad_W_out = DTensor.from_local(grad_W_local, ctx.W_s_mesh, ctx.W_s_placements)
        else:
            grad_W_out = grad_W_s32.to(W_s_full.dtype)

        return grad_h_out, grad_W_out, None, None, None, None, None


def streaming_forward_kl(h_t, W_t, h_s, W_s, mask_bt, temperature: float, vchunk: int):
    """
    Exact forward KL: KL( pt || ps ) with temperature, without materializing logits.
      pt = softmax((h_t W_t^T)/T), ps = softmax((h_s W_s^T)/T)
    mask_bt: (B, T) bool mask of positions to include (e.g. label != -100)
    Returns: scalar KL averaged over masked tokens.

    Uses a custom autograd.Function so peak memory is O(B*T*vchunk).
    When W_s / W_t are FSDP2 Shard(0) DTensors, all-gathers both to full vocab.
    """
    return _StreamingForwardKLFunc.apply(
        h_s, W_s, h_t, W_t, mask_bt, temperature, vchunk,
    )
