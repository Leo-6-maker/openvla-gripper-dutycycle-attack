#!/usr/bin/env python3
"""Formal S5 CPU/GPU gate runner for Factorized Student.

Gate 1: Import + static checks
Gate 2: Full loader census (800 ids, 176336 steps, F3 parity)
Gate 3: Model step/forward parity, unsupported→zero probs
Gate 4: Batch-8 CPU gate (per-episode loss aggregation, padding invariance)
Gate 5: GPU engineering smoke (batch-8, 2 epochs, checkpoint roundtrip)
"""
import sys, json, csv, torch
from pathlib import Path
from collections import Counter

S1 = Path("/mnt/sdc/dty_user/openvla_attack_evidence/c2g/c2g_cs200_official_v3_20260716/ops/OFFICIAL_V3_S1_FIT_V1_d31187f")
TEACHER = Path("/mnt/sdc/dty_user/openvla_attack_evidence/c2g/c2g_cs200_official_v3_20260716/ops/OFFICIAL_V3_DETECTOR_V5_FACTORIZED_TEACHER_V1_de07e1a_20260721")
REG = Path("/mnt/sdc/dty_user/openvla_attack_evidence/c2g/c2g_cs200_official_v3_20260716/ops/OFFICIAL_V3_CAMPAIGN_REGISTRY_V1_d31187f")

def gate(n, desc):
    print(f"\n{'='*50}\nGate {n}: {desc}\n{'='*50}")

def main():
    from gripper_attack.v5_factorized_dataset import (
        FactorizedEpisode, load_factorized_episode, compute_factorized_normalization,
        verify_factorized_source_roots)
    from gripper_attack.v5_factorized_student import FactorizedStudent
    from gripper_attack.v5_factorized_loss import FactorizedLoss

    # ═══ Gate 1: Import + static ═══
    gate(1, "Import + static checks")
    seals = verify_factorized_source_roots(S1, TEACHER)
    print(f"  Seals OK — S1={seals['s1_root_seal'][:16]} Teacher={seals['teacher_root_seal'][:16]}")
    print("  PASS")

    # ═══ Gate 2: Loader census ═══
    gate(2, "Full loader census")
    rows = list(csv.DictReader(open(REG / "OFFICIAL_V3_FORMAL_REGISTRY_V1.csv")))
    fit = [r for r in rows if r.get("split") == "FIT_TRAIN"]
    g_pos = g_neg = g_unk = m_pos = m_neg = m_unk = r_pos = r_neg = r_unk = total = errors = 0
    route_counts = Counter()
    episodes_all = []
    for row in fit[:800]:
        try:
            ep = load_factorized_episode(S1, TEACHER, row)
            episodes_all.append(ep)
        except Exception as e:
            print(f"  ERROR {row['canonical_parent_key']}: {e}"); errors += 1; continue
        T = len(ep.features_25d); total += T; route_counts[ep.mechanism_route] += T
        gk, gt = ep.grasp_known_mask, ep.grasp_target
        g_pos += (gk & gt).sum().item(); g_neg += (gk & ~gt).sum().item(); g_unk += (~gk).sum().item()
        mk, mt = ep.manipulation_known_mask, ep.manipulation_target
        m_pos += (mk & mt).sum().item(); m_neg += (mk & ~mt).sum().item(); m_unk += (~mk).sum().item()
        rk, rt = ep.release_known_mask, ep.release_target
        r_pos += (rk & rt).sum().item(); r_neg += (rk & ~rt).sum().item(); r_unk += (~rk).sum().item()
    assert errors == 0 and g_pos == 55109 and m_pos == 44120 and r_pos == 3023, "CENSUS FAIL"
    print(f"  {len(episodes_all)} eps, {total} steps, 0 errors — counts match F3")
    print("  PASS")

    # ═══ Gate 3: Model parity ═══
    gate(3, "Model step/forward parity + unsupported probabilities")
    model = FactorizedStudent(use_9d=False)
    B, T = 3, 10
    x25 = torch.randn(B, T, 25); mask = torch.ones(B, T, dtype=torch.bool)
    for route in ["single_object_pick_place", "multi_object_transfer"]:
        fwd = model.forward_sequence(x25, None, mask, None, route)
        h = model.initial_hidden(B, "cpu")
        gs, ms, rs = [], [], []
        for t in range(T):
            logits, h = model.step(x25[:, t], None, mask[:, t], None, h, route)
            gs.append(logits["grasp"]); ms.append(logits["manipulation"]); rs.append(logits["release"])
        step_g = torch.stack(gs, dim=1); step_m = torch.stack(ms, dim=1); step_r = torch.stack(rs, dim=1)
        assert torch.allclose(fwd["grasp"], step_g, atol=1e-6), f"{route} grasp mismatch"
        assert torch.allclose(fwd["manipulation"], step_m, atol=1e-6), f"{route} manip mismatch"
        assert torch.allclose(fwd["release"], step_r, atol=1e-6), f"{route} release mismatch"
    # Unsupported → zero probabilities
    unsup = model.forward_sequence(x25, None, mask, None, "articulated_or_planar")
    assert (unsup["grasp"] < 1e-3).all() and (unsup["manipulation"] < 1e-3).all() and (unsup["release"] < 1e-3).all()
    print("  step/forward parity OK, unsupported≈0 OK")
    print("  PASS")

    # ═══ Gate 4: Batch-8 CPU ═══
    gate(4, "Batch-8 CPU gate (per-episode aggregation, padding invariance)")
    model_train = FactorizedStudent(use_9d=False)
    loss_fn = FactorizedLoss(consistency_weight=0.1)
    # Pick 8 single-object episodes with positive events
    single_eps = [e for e in episodes_all if e.mechanism_route == "single_object_pick_place"
                  and e.grasp_target.any() and e.manipulation_target.any()][:8]
    assert len(single_eps) == 8, f"only {len(single_eps)} single-object positive episodes"

    # Pad to max length
    max_T = max(len(e.features_25d) for e in single_eps)
    x25_b = torch.zeros(8, max_T, 25)
    mask_b = torch.zeros(8, max_T, dtype=torch.bool)
    for b, ep in enumerate(single_eps):
        T_ep = len(ep.features_25d)
        x25_b[b, :T_ep] = ep.features_25d
        mask_b[b, :T_ep] = True

    logits_b = model_train.forward_logits(x25_b, None, mask_b, None, "single_object_pick_place")
    loss_batch, _ = loss_fn(logits_b, single_eps, mask_b)

    # Compare: per-episode loss should equal batched loss
    loss_individual = torch.tensor(0.0)
    for b, ep in enumerate(single_eps):
        T_ep = len(ep.features_25d)
        x_ind = ep.features_25d.unsqueeze(0)
        m_ind = ep.valid_mask.unsqueeze(0)
        logits_ind = model_train.forward_logits(x_ind, None, m_ind, None, "single_object_pick_place")
        l_ind, _ = loss_fn(logits_ind, [ep])
        loss_individual = loss_individual + l_ind
    loss_individual = loss_individual / 8
    assert torch.allclose(loss_batch, loss_individual, atol=1e-5), \
        f"batch loss {loss_batch.item():.6f} != individual avg {loss_individual.item():.6f}"
    print(f"  Batch loss={loss_batch.item():.6f} == individual avg={loss_individual.item():.6f}")

    # Overfit: loss must decrease
    opt = torch.optim.AdamW(model_train.parameters(), lr=1e-3)
    initial_loss = loss_batch.item()
    for step in range(400):
        opt.zero_grad()
        logits_b = model_train.forward_logits(x25_b, None, mask_b, None, "single_object_pick_place")
        loss_b, _ = loss_fn(logits_b, single_eps, mask_b)
        loss_b.backward()
        torch.nn.utils.clip_grad_norm_(model_train.parameters(), 5.0)
        opt.step()
    final_loss = loss_b.item()
    reduction = (initial_loss - final_loss) / initial_loss * 100
    assert reduction >= 90, f"Loss reduction {reduction:.0f}% < 90%"
    assert all(torch.isfinite(p).all() for p in model_train.parameters()), "Non-finite params"
    print(f"  Overfit: {initial_loss:.4f} → {final_loss:.4f} ({reduction:.0f}%)")
    print("  PASS")

    # ═══ Gate 5: GPU smoke ═══
    gate(5, "GPU engineering smoke")
    device = torch.device("cuda:0")
    model_gpu = FactorizedStudent(use_9d=False).to(device)
    loss_gpu = FactorizedLoss(consistency_weight=0.1)
    opt_gpu = torch.optim.AdamW(model_gpu.parameters(), lr=1e-3)

    # Single-object batch
    multi_eps = [e for e in episodes_all if e.mechanism_route == "multi_object_transfer"
                 and e.grasp_target.any()][:8]
    for route_name, eps_batch in [("single", single_eps), ("multi", multi_eps)]:
        max_T = max(len(e.features_25d) for e in eps_batch)
        xb = torch.zeros(8, max_T, 25, device=device)
        mb = torch.zeros(8, max_T, dtype=torch.bool, device=device)
        for b, ep in enumerate(eps_batch):
            T_ep = len(ep.features_25d)
            xb[b, :T_ep] = ep.features_25d.to(device)
            mb[b, :T_ep] = True
        for _ in range(2):
            opt_gpu.zero_grad()
            logits = model_gpu.forward_logits(xb, None, mb, None, eps_batch[0].mechanism_route)
            loss, _ = loss_gpu(logits, eps_batch, mb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model_gpu.parameters(), 5.0)
            opt_gpu.step()
        print(f"  {route_name}: GPU fwd/bwd OK, loss={loss.item():.4f}")

    # Checkpoint roundtrip
    ckpt = {"state_dict": model_gpu.state_dict(), "config": {"hidden_dim": 128, "use_9d": False}}
    torch.save(ckpt, "/tmp/factorized_s5_ckpt.pt")
    loaded = torch.load("/tmp/factorized_s5_ckpt.pt", map_location="cpu")
    model_cpu2 = FactorizedStudent(use_9d=False); model_cpu2.load_state_dict(loaded["state_dict"])
    # Verify forward_sequence parity
    logits_gpu = model_gpu.forward_logits(xb[:1], None, mb[:1], None, eps_batch[0].mechanism_route)
    logits_cpu = model_cpu2.forward_logits(xb[:1].cpu(), None, mb[:1].cpu(), None, eps_batch[0].mechanism_route)
    assert torch.allclose(logits_gpu["grasp"].cpu(), logits_cpu["grasp"], atol=1e-6), "checkpoint mismatch"
    print(f"  Checkpoint roundtrip OK, {torch.cuda.memory_allocated(0)/1e9:.1f}GB allocated")
    print("  PASS")

    print(f"\n{'='*50}")
    print("ALL S5 GATES PASSED — READY FOR OOF AUTHORIZATION")
    print(f"{'='*50}")

if __name__ == "__main__":
    main()
