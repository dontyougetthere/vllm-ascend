# SPDX-License-Identifier: Apache-2.0
"""Small A5 numeric/capture checks; no model download or full-model serving.

Pure dispatch/schema/guard cases belong in CPU UT. These cases need actual
NPU formats, zero-copy views, graph replay and independent tensor references.
Full stress/capacity matrices remain in nightly tests; reuse their helpers.
"""

import pytest
import torch

from tests.e2e.nightly.single_node.ops.singlecard_ops import test_kimi_k3_dspark_hp_c8_multiquery as metadata_cases
from tests.e2e.nightly.single_node.ops.singlecard_ops.triton import (
    test_mxfp_high_precision_attention as attention_cases,
)


@pytest.mark.parametrize("capacity", [1024, 32768])
def test_mxfp8_attention_matches_fp32_reference_and_replay(capacity):
    attention_cases.test_reader_fp32_reference_and_dynamic_graph(capacity, torch.float32)


def test_pp2_causal_lengths_and_stable_buffers():
    metadata_cases.test_pp2_multiquery_reuses_causal_metadata_and_fixed_addresses()


@pytest.mark.parametrize("latent,rope_width", [(512, 64), (256, 32)])
@torch.inference_mode()
def test_current_cache_packet_bytes_and_dynamic_graph(latent, rope_width):
    """Transport-only layouts; extra widths do not enable new model backends."""
    from vllm_ascend.attention.context_parallel.kimi_c8_history_current import (
        KimiC8CurrentWorkspace,
        assemble_current_cache_bytes,
        extract_current_cache_bytes,
    )

    torch.npu.set_device(0)
    cpu_key = (torch.arange(48 * latent).reshape(3, 1, 16, latent) % 7).to(torch.float8_e4m3fn)
    cpu_rope = (torch.arange(48 * rope_width).reshape(3, 1, 16, rope_width) % 13).to(torch.bfloat16)
    key = cpu_key.to("npu")
    rope = cpu_rope.to("npu")
    slots = torch.tensor([0, 17, -1, 48], dtype=torch.int64, device="npu")
    workspace = KimiC8CurrentWorkspace(4, "npu", kv_lora_rank=latent, qk_rope_head_dim=rope_width, dcp_size=2)
    packets = workspace.gathered_packets.view(2, 4, workspace.packet_bytes)
    owners = workspace.gathered_owners.view(2, 4)

    def transport():
        extract_current_cache_bytes(key, rope, slots, workspace.local_packet, workspace.local_owners)
        packets[0].copy_(workspace.local_packet)
        packets[1].zero_()
        owners[0].copy_(workspace.local_owners)
        owners[1].zero_()
        assemble_current_cache_bytes(packets, owners, workspace.packet, workspace.counts)

    stream = torch.npu.Stream()
    stream.wait_stream(torch.npu.current_stream())
    with torch.npu.stream(stream):
        for _ in range(3):
            transport()
    torch.npu.current_stream().wait_stream(stream)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, stream=stream):
        transport()
    pointers = (workspace.packet.data_ptr(), workspace.counts.data_ptr())
    for positions in ([0, 17, -1, 48], [47, 16, 0, -1]):
        slots.copy_(torch.tensor(positions, dtype=torch.int64))
        graph.replay()
        torch.npu.synchronize()
        expected = torch.zeros((4, workspace.packet_bytes), dtype=torch.uint8)
        for row, position in enumerate(positions):
            if 0 <= position < 48:
                expected[row] = torch.cat(
                    (
                        cpu_key.reshape(48, latent)[position].view(torch.uint8),
                        cpu_rope.reshape(48, rope_width)[position].view(torch.uint8),
                    )
                )
        torch.testing.assert_close(workspace.packet.cpu(), expected, rtol=0, atol=0)
        assert workspace.counts.cpu().tolist() == [int(0 <= position < 48) for position in positions]
        assert pointers == (workspace.packet.data_ptr(), workspace.counts.data_ptr())
    # Duplicate ownership must remain visible, never silently merge payloads.
    owners[1].copy_(owners[0])
    assemble_current_cache_bytes(packets, owners, workspace.packet, workspace.counts)
    torch.npu.synchronize()
    assert workspace.counts.cpu().tolist() == [2, 2, 2, 0]
    assert torch.count_nonzero(workspace.packet).item() == 0
