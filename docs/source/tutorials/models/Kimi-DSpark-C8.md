# Kimi DSpark C8 QuantFlashMLA validation

This release-branch port carries the QuantFlashMLA integration from upstream
PR #21569 and the follow-up fixes described below. It does not add a native
operator or require users to set a special cache block size.

## Behavior

- Resolve the DSpark draft's own quantization recipe and layer namespace,
  independently of the target model. Unquantized drafts retain floating-point
  cache storage.
- Identify both MLA projection weight methods before fused postprocessing.
  FP8 attention caches do not imply quantized linear weights.
- Quantize the full query in the existing cache units: `[latent, rope / k_scale]`.
  The existing dynamic-quant operator receives the model floating dtype and
  supplies a common query descale for the FP8 query.
- Prepare explicit writer scale vectors and scatter the quantized staging rows,
  rather than the writer's unquantized optional outputs.
- Allocate dense FP8 staging using the backend's supported kernel page geometry,
  independently of the provisional manager block size. Actual cache writes use
  the original slot mapping. The global cache configuration is not modified.

## Version and validation boundary

The hardware evidence below belongs to the isolated R5 development snapshot
based on upstream PR #21569 at `51a96c8180c33f249fd77f64f71201e5f7180044`,
paired with vLLM `24420b7a341af37a9bc3186e6fc6062512e0aa58`.
Its MLA source SHA256 was
`3d606c53c2dff5948395f4316c2ba3aaa94470940ec7146e41ed99f4c05b4452`.

This PR instead targets `releases/v0.32.0rc` at
`b12b6f7f47bb0b87ae3268790f1b2503e95e0e01`. The release's vLLM dependency is
v0.32.0. In particular, this port uses the owning layer's draft marker because
the release KV cache specification has no `is_draft` field. Existing release
floating FlashMLA eligibility and cache-planning behavior are preserved. A
layer-local QuantFlashMLA draft can opt into fused raw-cache allocation even
when its surrounding target configuration selects SFA. The main-only
paired-SFA allocator test is not imported into a release without that allocator.

**The release port has not been rerun on NPU. Historical results are not a
claim that this new revision, a different dependency pair, or release CI passed.**

## Historical R5 hardware results

Environment: one Ascend 950 DT host, two NPUs, TP2, fixed A5 RC1 image,
eager execution, standard verification, greedy sampling, seed 20261116,
no entropy/relaxed verification, no synthetic or adaptive acceptance.
The image ID was
`sha256:04c836f810f42a03639e7831a1b1fedc51f1375cb55f9c2626d21cd0b70e5c7a`.

The reduced target retained 93 layers but only eight experts with top-k eight.
The five-layer DSpark draft retained the original target feature-layer IDs.
All three configurations omitted `block_size`; each ran the same English and
Chinese short prompt twice with a maximum of 16 generated tokens.

| Configuration | Result | Accepted/proposed draft tokens | Acceptance rate | Mean accepted prefix |
| --- | --- | --- | --- | --- |
| Target only | Four completions | Not applicable | Not applicable | Not applicable |
| BF16 DSpark | All tokens matched target only | 58/168 | 34.5238% | 2.4167 |
| C8 DSpark | All tokens matched target only | 56/182 | 30.7692% | 2.1538 |

Each C8 rank completed 160 actual QuantFlashMLA calls across its five draft
layers. Queries, descales and outputs were finite, without backend fallback.
The allocation-time manager block remained 16 while the staging page used the
backend-supported geometry. Additional explicit-block-size coverage passed.
Acceptance counts include the final verification round before completion-length
truncation; they are not the number of tokens returned to the client.

Seventeen actual-module CPU/mock workspace tests passed. Eight real writer cases
covered empty inputs, page boundaries, reversed slots, capacity and overflow
rejection. Staging/cache contents and locations matched, unwritten rows remained
unchanged, and optional BF16 returns matched a separate floating writer control.
The evidence archive SHA256 is
`44fa855dc4c6187d5ad0b6808e9c240c75696b7135bc03c4302f998f9c65e37e`.
Raw model data, deployment endpoints and logs are retained privately.

## Interpretation and remaining validation

Draft hidden-state differences are not a target-quality acceptance criterion.
Under a correct standard verifier, draft quantization is primarily evaluated
through proposal quality, acceptance length and performance. The short greedy
checks above establish only their observed output equality and execution path.

The reduced target generated repetitive text. These results do not establish
semantic quality, representative acceptance rates, full-expert model quality,
random-sampling distribution equivalence, long-context behavior, graph replay,
context-parallel or disaggregated execution, or performance. Diagnostic
synchronization makes the recorded run times unsuitable for benchmarking.

For this release port, run focused tests including
`test_quant_flashmla.py`, `test_quant_flashmla_writer_workspace.py`,
`test_dspark_c8_contracts.py`, the affected quantization tests and MRV2 cache-spec
tests, then repeat the default-configuration three-way NPU comparison with the
release's matching vLLM dependency. Do not substitute an explicit block size to
hide a default-configuration failure.
