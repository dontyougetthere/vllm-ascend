# SPDX-License-Identifier: Apache-2.0
"""Private K3 dual-C8 DCP history/current adapter; PCP remains unsupported."""

from dataclasses import dataclass

import torch
import torch.distributed as dist
from vllm.triton_utils import tl, triton
from vllm.v1.attention.backends.utils import get_dcp_local_seq_lens

from vllm_ascend.ops.triton.mla_high_precision_decode import decode


@triton.jit
def _extract(
    K,
    R,
    SLOTS,
    PACKET,
    OWNERS,
    KB: tl.constexpr,
    KT: tl.constexpr,
    RB: tl.constexpr,
    RT: tl.constexpr,
    BS: tl.constexpr,
    NPAGES: tl.constexpr,
    LATENT_BYTES: tl.constexpr,
    PACKET_BYTES: tl.constexpr,
    TILE: tl.constexpr,
):
    t = tl.program_id(0)
    slot = tl.load(SLOTS + t).to(tl.int64)
    valid = (slot >= 0) & (slot < NPAGES * BS)
    page = tl.where(valid, slot // BS, 0)
    offset = tl.where(valid, slot % BS, 0)
    d = tl.arange(0, TILE)
    k = tl.load(K + page * KB + offset * KT + d, valid & (d < LATENT_BYTES), other=0)
    r = tl.load(
        R + page * RB + offset * RT + (d - LATENT_BYTES), valid & (d >= LATENT_BYTES) & (d < PACKET_BYTES), other=0
    )
    value = tl.where(d < LATENT_BYTES, k, r)
    tl.store(PACKET + t * PACKET_BYTES + d, value, d < PACKET_BYTES)
    tl.store(OWNERS + t, valid.to(tl.int32))


@triton.jit
def _assemble(
    PACKETS, OWNERS, OUT, COUNTS, T: tl.constexpr, CP: tl.constexpr, PACKET_BYTES: tl.constexpr, TILE: tl.constexpr
):
    t = tl.program_id(0)
    d = tl.arange(0, TILE)
    value = tl.full((TILE,), 0, tl.int32)
    count = tl.full((), 0, tl.int32)
    for rank in range(CP):
        owner = tl.load(OWNERS + rank * T + t)
        byte = tl.load(PACKETS + (rank * T + t) * PACKET_BYTES + d, d < PACKET_BYTES, other=0).to(tl.int32)
        value = value + tl.where(owner == 1, byte, 0)
        count = count + owner
    # Bad ownership is observable in COUNTS and produces a neutral packet,
    # rather than silently combining two byte streams into valid-looking data.
    value = tl.where(count == 1, value, 0).to(tl.uint8)
    tl.store(OUT + t * PACKET_BYTES + d, value, d < PACKET_BYTES)
    tl.store(COUNTS + t, count)


@triton.jit
def _assert_current_owners(COUNTS, LENGTHS):
    row = tl.program_id(0)
    count = tl.load(COUNTS + row)
    live = tl.load(LENGTHS + row) > 0
    tl.device_assert(count == live.to(tl.int32), "K3 C8 current cache ownership must be unique for live rows")


def assert_current_owners(counts, lengths):
    assert counts.ndim == lengths.ndim == 1 and counts.shape == lengths.shape
    assert counts.is_contiguous() and lengths.is_contiguous()
    assert counts.device == lengths.device and counts.dtype == lengths.dtype == torch.int32
    if counts.numel():
        # Explicit debug=True is required: never depend on TRITON_DEBUG to
        # enable this correctness guard. Remains device-only during capture.
        _assert_current_owners[(counts.numel(),)](counts, lengths, debug=True)


def extract_current_cache_bytes(key, rope, slots, packet, owners):
    assert key.ndim == 4 and key.shape[1] == 1 and key.shape[-1] > 0
    assert rope.ndim == 4 and rope.shape[:-1] == key.shape[:-1] and rope.shape[-1] > 0
    assert key.dtype == torch.float8_e4m3fn and rope.dtype == torch.bfloat16
    assert key.stride(-1) == rope.stride(-1) == 1
    assert slots.ndim == 1 and slots.dtype in (torch.int32, torch.int64) and slots.is_contiguous()
    n = slots.numel()
    latent_bytes = key.shape[-1] * key.element_size()
    packet_bytes = latent_bytes + rope.shape[-1] * rope.element_size()
    assert packet.shape == (n, packet_bytes) and packet.dtype == torch.uint8 and packet.is_contiguous()
    assert owners.shape == (n,) and owners.dtype == torch.int32 and owners.is_contiguous()
    assert all(t.device == key.device for t in (rope, slots, packet, owners))
    raw_rope = rope.view(torch.uint8)
    if n:
        _extract[(n,)](
            key.view(torch.uint8),
            raw_rope,
            slots,
            packet,
            owners,
            key.stride(0),
            key.stride(2),
            raw_rope.stride(0),
            raw_rope.stride(2),
            key.shape[2],
            key.shape[0],
            latent_bytes,
            packet_bytes,
            triton.next_power_of_2(packet_bytes),
        )


def assemble_current_cache_bytes(packets, owners, packet, counts):
    assert packets.ndim == 3 and packets.shape[-1] > 0
    cp, n, packet_bytes = packets.shape
    assert cp >= 1 and packets.dtype == torch.uint8 and packets.is_contiguous()
    assert owners.shape == (cp, n) and owners.dtype == torch.int32 and owners.is_contiguous()
    assert packet.shape == (n, packet_bytes) and packet.dtype == torch.uint8 and packet.is_contiguous()
    assert counts.shape == (n,) and counts.dtype == torch.int32 and counts.is_contiguous()
    assert all(t.device == packets.device for t in (owners, packet, counts))
    if n:
        _assemble[(n,)](packets, owners, packet, counts, n, cp, packet_bytes, triton.next_power_of_2(packet_bytes))


def use_kimi_c8_dcp_current(config):
    model = getattr(getattr(config, "model_config", None), "hf_text_config", None)
    spec = getattr(config, "speculative_config", None)
    parallel = getattr(config, "parallel_config", None)
    cache = getattr(config, "cache_config", None)
    return (
        getattr(model, "model_type", None) == "kimi_linear"
        and getattr(spec, "method", None) == "dspark"
        and getattr(spec, "kv_cache_dtype", None) == "mxfp8"
        and getattr(cache, "cache_dtype", None) == "fp8"
        and getattr(parallel, "decode_context_parallel_size", 1) == 2
        and getattr(parallel, "prefill_context_parallel_size", 1) == 1
        and getattr(parallel, "pipeline_parallel_size", 1) in (1, 2)
    )


def query_layout(query_lengths, live_requests, actual_tokens, max_queries):
    if not 0 <= live_requests <= len(query_lengths):
        raise ValueError("K3 C8 live request count exceeds query lengths")
    live = query_lengths[:live_requests]
    if any(not isinstance(q, int) or isinstance(q, bool) or not 1 <= q <= max_queries for q in live):
        raise ValueError("K3 C8 requires bounded positive query lengths")
    if sum(live) != actual_tokens:
        raise ValueError("K3 C8 actual token count differs from live query lengths")
    requests = []
    ends = []
    starts = []
    cursor = 0
    for req, n in enumerate(live):
        requests.extend([req] * n)
        ends.extend(range(1, n + 1))
        starts.extend([cursor] * n)
        cursor += n
    return live, requests, ends, starts


def causal_local_lengths(global_history, requests, causal_counts, *, dcp_size, dcp_rank, interleave):
    """Partition each query's global causal endpoint, not its whole verify chunk."""
    global_causal = global_history.index_select(0, requests) + causal_counts
    return get_dcp_local_seq_lens(
        global_causal,
        dcp_size=dcp_size,
        dcp_rank=dcp_rank,
        cp_kv_cache_interleave_size=interleave,
    )


@dataclass
class KimiC8CurrentMetadata:
    history_lengths: torch.Tensor
    history_pages: torch.Tensor
    current_lengths: torch.Tensor
    current_pages: torch.Tensor
    slots: torch.Tensor
    causal_lengths: torch.Tensor


class KimiC8MetadataBuffers:
    def __init__(self, capacity, columns, max_queries, device):
        self.capacity = capacity
        self.columns = columns
        self.max_queries = max_queries
        self.history_lengths = torch.empty(capacity, dtype=torch.int32, device=device)
        self.history_pages = torch.empty((capacity, columns), dtype=torch.int32, device=device)
        self.current_lengths = torch.empty(capacity, dtype=torch.int32, device=device)
        self.current_pages = torch.empty((capacity, max_queries), dtype=torch.int32, device=device)
        self.slots = torch.empty(capacity, dtype=torch.int64, device=device)
        self.causal_lengths = torch.empty(capacity, dtype=torch.int32, device=device)

    def refresh(self, builder, common):
        rows = max(builder.num_decode_tokens, builder.graph_pad_size)
        if rows > self.capacity or builder.block_table.shape[1] != self.columns:
            raise ValueError("K3 C8 metadata exceeds retained shape")
        live_count = min(builder.num_decodes, builder._hp_c8_actual_reqs)
        lengths = (
            common.query_start_loc_cpu[1 : builder.num_decodes + 1] - common.query_start_loc_cpu[: builder.num_decodes]
        ).tolist()
        live, requests, ends, starts = query_layout(lengths, live_count, builder.num_decode_tokens, self.max_queries)
        source = common.seq_lens
        if source is None or source.device != builder.block_table.device or source.device.type == "cpu":
            raise ValueError("K3 C8 DCP history requires runner device sequence lengths")
        if common.slot_mapping.numel() < builder.num_decode_tokens:
            raise ValueError("K3 C8 DCP current slots are incomplete")
        result = KimiC8CurrentMetadata(
            self.history_lengths[:rows],
            self.history_pages[:rows],
            self.current_lengths[:rows],
            self.current_pages[:rows],
            self.slots[:rows],
            self.causal_lengths[:rows],
        )
        result.history_lengths.zero_()
        result.history_pages.zero_()
        result.causal_lengths.zero_()
        result.current_lengths.zero_()
        result.current_pages.zero_()
        result.slots.fill_(-1)
        if getattr(builder, "_kimi_c8_capture_dummy", False):
            # Native v2 capture uses PAD_SLOT_ID for every dummy token: these
            # rows own no cache. Capture all fixed-shape kernels/collectives
            # with inactive device lengths, then refresh these same buffers
            # normally before real replay. Never infer this from runtime slots;
            # ordinary missing ownership must still trip the device guard.
            return result
        active = builder.num_decode_tokens
        if active:
            device = source.device
            req = torch.tensor(requests, dtype=torch.int64, device=device)
            qlens = torch.tensor(live, dtype=torch.int32, device=device)
            # Subtract globally on authoritative device state, then partition.
            global_history = (source[:live_count] - qlens).clamp(min=0)
            local = get_dcp_local_seq_lens(
                global_history,
                dcp_size=builder.dcp_size,
                dcp_rank=builder.dcp_rank,
                cp_kv_cache_interleave_size=builder.cp_local_block_size,
            )
            result.history_lengths[:active].copy_(local.index_select(0, req))
            result.history_pages[:active].copy_(builder.block_table.index_select(0, req))
            result.current_lengths[:active].copy_(torch.tensor(ends, dtype=torch.int32, device=device))
            result.causal_lengths[:active].copy_(
                causal_local_lengths(
                    global_history,
                    req,
                    result.current_lengths[:active],
                    dcp_size=builder.dcp_size,
                    dcp_rank=builder.dcp_rank,
                    interleave=builder.cp_local_block_size,
                )
            )
            start = torch.tensor(starts, dtype=torch.int32, device=device)
            result.current_pages[:active].copy_(
                start[:, None] + torch.arange(self.max_queries, device=device, dtype=torch.int32)[None, :]
            )
            result.slots[:active].copy_(common.slot_mapping[:active])
        return result


def prepare_kimi_c8_metadata(builder, common):
    max_queries = 1 + (builder.vllm_config.speculative_config.num_speculative_tokens or 0)
    buffers = getattr(builder, "_kimi_c8_metadata_buffers", None)
    if buffers is None:
        buffers = KimiC8MetadataBuffers(
            builder._hp_c8_capacity, builder.block_table.shape[1], max_queries, builder.block_table.device
        )
        builder._kimi_c8_metadata_buffers = buffers
    elif buffers.max_queries != max_queries:
        raise ValueError("K3 C8 speculative width changed after metadata allocation")
    return buffers.refresh(builder, common)


class KimiC8CurrentWorkspace:
    def __init__(self, capacity, device, *, kv_lora_rank, qk_rope_head_dim, dcp_size):
        # Derive transport storage from the actual mixed-cache tensor layout.
        # This byte transport does not widen the attention backend's shape guard.
        dimensions = (capacity, kv_lora_rank, qk_rope_head_dim, dcp_size)
        if any(not isinstance(value, int) or isinstance(value, bool) or value <= 0 for value in dimensions):
            raise ValueError("C8 current workspace dimensions must be positive integers")
        self.capacity = capacity
        self.dcp_size = dcp_size
        self.packet_bytes = kv_lora_rank * torch.float8_e4m3fn.itemsize + qk_rope_head_dim * torch.bfloat16.itemsize
        self.local_packet = torch.empty((capacity, self.packet_bytes), dtype=torch.uint8, device=device)
        self.local_owners = torch.empty(capacity, dtype=torch.int32, device=device)
        self.gathered_packets = torch.empty((dcp_size * capacity, self.packet_bytes), dtype=torch.uint8, device=device)
        self.gathered_owners = torch.empty(dcp_size * capacity, dtype=torch.int32, device=device)
        self.packet = torch.empty_like(self.local_packet)
        self.counts = torch.empty_like(self.local_owners)


def forward_kimi_c8_dcp_current(impl, preprocessed, block_size, metadata):
    current = metadata.decode.kimi_c8_current
    workspace = getattr(impl, "_kimi_c8_current_workspace", None)
    if current is None or workspace is None:
        raise RuntimeError("K3 C8 DCP requires prepared current metadata and workspace")
    q = preprocessed.ql_nope
    qr = preprocessed.q_pe
    qs = preprocessed.dequant_scale_q_nope
    rows = q.shape[0]
    if rows > workspace.capacity or current.slots.numel() != rows or workspace.dcp_size != impl.dcp_size:
        raise ValueError("K3 C8 DCP current rows exceed retained workspace")
    if (
        impl.dcp_size != 2
        or impl.pcp_enabled
        or impl.num_kv_heads != 1
        or impl.kv_lora_rank != 512
        or impl.qk_rope_head_dim != 64
    ):
        raise ValueError("K3 C8 DCP current geometry is outside its guarded contract")
    if qs is None or qs.shape != q.shape[:2] or q.shape[1] != impl.num_heads * impl.dcp_size:
        raise ValueError("K3 C8 DCP requires gathered query heads and per-head descales")
    key = preprocessed.k_nope.view(-1, 1, block_size, impl.kv_lora_rank)
    rope = preprocessed.k_pe.view(-1, 1, block_size, impl.qk_rope_head_dim)
    packet = workspace.local_packet[:rows]
    owners = workspace.local_owners[:rows]
    extract_current_cache_bytes(key, rope, current.slots, packet, owners)
    gathered_packets = workspace.gathered_packets[: impl.dcp_size * rows]
    gathered_owners = workspace.gathered_owners[: impl.dcp_size * rows]
    dist.all_gather_into_tensor(gathered_packets, packet, group=impl.dcp_group.device_group)
    dist.all_gather_into_tensor(gathered_owners, owners, group=impl.dcp_group.device_group)
    exact_packet = workspace.packet[:rows]
    counts = workspace.counts[:rows]
    assemble_current_cache_bytes(
        gathered_packets.view(impl.dcp_size, rows, workspace.packet_bytes),
        gathered_owners.view(impl.dcp_size, rows),
        exact_packet,
        counts,
    )
    assert_current_owners(counts, current.current_lengths)
    # Preserve the unique-owner assertion above. All current KV bytes already
    # reside in these pages; each query reads only its own causal prefix.
    # Use the same reduction path as ordinary single-token C8 decode so the
    # verify chunk does not introduce an extra FP32 merge before BF16 V-up.
    full_output, full_lse = decode(
        q,
        qr,
        key,
        rope,
        qs.contiguous(),
        impl.fak_descale_float,
        impl.fak_descale_float,
        current.history_pages,
        current.causal_lengths,
        impl.scale,
        out_dtype=torch.float32,
    )
    return impl._merge_dcp_attention_output(full_output, full_lse)
