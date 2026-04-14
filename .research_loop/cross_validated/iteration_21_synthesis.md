# Iteration 21 — Research Synthesis: Selective Segment Compression for Race Time Recovery

## Bottleneck
Race time 14.10s exceeds aspirational <14s target. Need to recover ~0.10-0.15s without regressing tracking accuracy (avg 0.218m).

## Papers Analyzed (New)
1. **STORM: Spatial-Temporal Iterative Optimization** (Zhang et al., HIT Shenzhen, 2025, arXiv:2503.03252) — Spatial-temporal decoupling framework alternating QP (path shape) + LP (segment times). Key: per-segment time optimization via LP is tractable and the guidance gradient prevents over-conservative timing.
2. **Sequence Modeling for Time-Optimal TOPP** (Mao et al., UPenn, CoRL 2025, arXiv:2506.13915) — LSTM encoder-decoder predicts per-segment speed from geometry. Key: per-segment speed has strong sequential dependencies (causal left-to-right).

## Papers Referenced (Existing)
3. **FBGA** (Piazza 2025) — Forward-backward velocity profiling matches OCP within 0.36%. Segment-wise piecewise-constant acceleration. Key insight: the forward-backward scheme naturally compresses easy segments more.
4. **TOPPQuad** (Mao, IROS 2024) — Fix geometry, optimize speed → 40-50% faster. Key: geometric path is fixed, only timing changes.

## Research Consensus

### 1. Spatial-temporal decoupling is the right framework
All four papers agree: fix the geometric path first, then optimize timing separately. Our system already does this (L-BFGS path → TOPP retime). The opportunity is in the TOPP retimer's compression floor.

### 2. Per-segment timing should be individually optimized, not uniformly constrained
- STORM: LP solves for each segment's knot interval independently (within coupling constraints)
- FBGA: forward-backward yields different speeds at each segment based on local curvature
- Sequence Modeling: per-segment speed prediction captures local geometry
- **Our system's gap**: We use a UNIFORM `max_compression = 0.68` floor for ALL segments. This means easy straight segments (gate-1 approach, gate-5-6 connector) are limited to 68% of L-BFGS time even when TOPP analysis says they could go faster.

### 3. Sequential coupling matters — but only limits turns, not straights
The LSTM paper shows speed at segment k depends on k-1 (causal). FBGA's forward-backward naturally handles this. The implication: straight segments CAN be compressed more because the forward-backward propagation naturally slows at high-curvature segments and speeds up at low-curvature. The uniform floor is what's preventing straights from realizing their speed potential.

## Contradictions
None found on this specific topic. All papers support segment-selective timing.

## Proposed Implementation Direction

**Segment-selective max_compression floor in `_topp_retime()`:**
- S-turn segments (in `s_turn_segments` set): keep `max_compression = 0.68`
- High-curvature segments (near helix gates with proximity < 6m): keep `max_compression = 0.68`
- All other segments (straights, shallow curves): lower to `max_compression = 0.60`

**Evidence strength**: Strong. FBGA and STORM both show that easy segments should be compressed more. The uniform floor was a simplification that protected helix/S-turn regions but penalized straight segments.

**Risk assessment**: Low. We're only relaxing the floor on segments that already have low tracking error (0.112-0.158m). The S-turn and helix regions retain their protective floor.

**Expected outcome**: Race time 14.10 → ~13.95s (recovery of ~0.15s from straight segment compression). Tracking accuracy should be maintained since the affected segments already have excellent tracking.
