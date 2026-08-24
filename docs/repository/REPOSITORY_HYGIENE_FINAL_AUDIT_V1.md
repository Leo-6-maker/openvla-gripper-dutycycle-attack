# Repository hygiene and paper-support final audit V1

Status: `REPOSITORY_HYGIENE_PAPER_SUPPORT_READY_FOR_PI_REVIEW`

Audit date: 2026-08-24

Scope:

- repository: `Leo-6-maker/openvla-gripper-dutycycle-attack`
- PR: `#136` (`OPEN`, `DRAFT`, `MERGEABLE` at the audited checkpoint)
- branch: `codex/repository-hygiene-paper-support-20260824`
- verified start HEAD: `544307997f7026e5b7b04e1400bbc675a3d63e2e`
- implementation/export checkpoint: `cdddba8888614abf7fe07f925855d37a0bc0f43a`
- deterministic export source HEAD: `a3a1846939df4be4c04de5c1de8bfee541b4d656`

This is a repository/software audit, not a scientific gate, rerun, or result.
No action in CODE-R0--R6 authorizes model loading, GPU work, simulator use,
`env.step`, adversarial generation, physical intervention, `V_phys`,
Eval160/protected reads, Stage Z Z1, or reopening F1/F1-D/BRIDGE.

## Final decision

CODE-R0 through CODE-R6 pass their repository-hygiene and paper-support gates.
No mandatory early-stop condition was encountered. The branch is ready for PI
review and must stop here without automatic merge.

## Phase checkpoints

| Phase | Commit(s) | Result |
| --- | --- | --- |
| CODE-R0 | `ed2b8ef9afcf0d48bd5ca3ccf9edfe0dec8b2a45`, `a8f5a068e58f2aeef735d9e39966da8cffed6ce6` | Repository map, lifecycle ledger, top-level clutter audit, and stale Stage Z snapshot classification. |
| CODE-R1 | `8ca76bbd8eed9841bf9264021d68a7ee19592779` | Immutable authority registry and read-only firewall audit. |
| CODE-R2 | `40ba12c32d70aca52a09d0ff0d2e222b9196c003` | Active code surface, reproducibility entry points, family indices, and read-only Paper V1 claim check. |
| CODE-R3 | `dd6d67734d7b65aa69cac7df9fbed8ab3e645366` | Root navigation rewritten around current authority and claim boundaries. |
| CODE-R4 | `4c6e7083cdb1017be8df0c8ef26c938497aa5775` | CPU/static CI matrix and all-PR repository-hygiene workflow. |
| CODE-R5 | `88bd40b0dd72ba061a3499a9c125fe8b5d65569f`, `ee374050bf247ed86d6c017d5306586268b3880e`, `2c83717e97b4a9ede9c46fa31e07ac327e98d6bf`, `a3a1846939df4be4c04de5c1de8bfee541b4d656`, `cdddba8888614abf7fe07f925855d37a0bc0f43a` | Deterministic sealed-data exporter, explicit claim-ID and digest checks, minimal CSV/JSON/TeX products, manifest, tests, CI, and navigation. |

## Authority integrity

PASS evidence:

- `python scripts/repository/audit_immutable_authority_paths.py` returned
  `CODE_R1_AUTHORITY_FIREWALL_PASS entries=88 sidecar_pairs=14 git_objects=2`.
- All 86 registered working-tree authority paths have the same Git blob at the
  verified start HEAD and the implementation/export checkpoint. The two
  historical Git-object authorities remain resolvable.
- All 20 `paper/PAPER_V1_*` and `paper/data/PAPER_V1_*` Git blobs are unchanged
  from the verified start HEAD.
- No delete or rename appears between the verified start HEAD and the audited
  checkpoint. Cleanup used indexing, lifecycle labels, and additive tooling.
- No Paper V1, Stage X/F1, or Stage Z sealed authority path was modified, moved,
  or deleted. Existing root seals, manifests, claim ledgers, and handoffs remain
  the authority source rather than filename inference.
- `python scripts/paper/check_paper_v1_claims.py` returned
  `PAPER_V1_CLAIM_AUDIT_READ_ONLY_PASS claims=48 eval160=UNREAD protected=UNREAD`.

## Software integrity

PASS evidence:

- Combined relevant CPU/static suite: `88 passed, 1 skipped` for repository,
  Paper V2, Stage X, the Stage X primary matrix runner, and Stage Z preparation.
- Repository navigation suite: `5 passed`.
- New repository/Paper V2 scripts and tests compile successfully.
- Format validation parsed eight JSON documents, fourteen CSV data rows, and
  nineteen dependency-free TeX commands.
- No compatibility move was performed, so no import path was displaced. The
  active Stage X suite and broad all-PR CI provide regression coverage for the
  one stale assertion corrected in CODE-R2.
- Patch whitespace checks pass.

Remote CI at `cdddba8888614abf7fe07f925855d37a0bc0f43a`:

- `repository-hygiene-cpu`: PASS, run `32662054703`
- `detector-v5-cpu`: PASS, run `32662054731`
- `source-registry`: PASS, run `32662054716`
- `stageb-cpu`: PASS, run `32662054744`

These CI results establish software consistency only; they do not promote a
scientific claim or authorize execution.

## Documentation integrity

PASS evidence:

- The root README states the current Paper V1, terminal Stage X/F1, and Stage Z
  HOLD statuses and points to their controlling seals/maps.
- `ACTIVE_CODE_SURFACE_V1.md` accounts for every tracked core source module and
  labels current, compatibility, historical, and sealed producer surfaces.
- `REPRODUCIBILITY_ENTRYPOINTS_V1.md` provides only CPU/static canonical entry
  points and records the Paper V2 deterministic check.
- Historical and legacy producers remain explicit and noncanonical; none was
  moved merely for cleanup.
- The main repository is documented as scientific source-of-truth and export
  owner. `Leo-6-maker/stage_aware_attack_on_vision_language_model` is documented
  as the presentation/LaTeX consumer, not a new evidence authority.

## Paper-export integrity

PASS evidence:

- `python scripts/paper_v2/export_paper_v2_evidence.py --check` returned
  `PAPER_V2_EXPORT_CHECK_PASS source_head=a3a1846939df4be4c04de5c1de8bfee541b4d656 files=8`.
- The manifest binds 14 exact authority inputs, seven generated products, the
  generator Git blob, source HEAD/tree/timestamp, and every generated SHA256.
- The bundle contains five complete JSON products, one tidy plot CSV, one TeX
  core-number macro file, and the manifest. The CSV and TeX are derived from
  the same in-memory JSON payloads; no scientific number is copied by hand.
- JSON rows retain source-defined primary units, denominators/censoring,
  statuses, claim IDs, authority paths, and digest/source bindings.
- X0 keeps source-defined T3/T5/T10 denominators and no iid uncertainty. Timing
  rows retain censored/non-identifiable cells without imputation. Stage IX
  retains the unsatisfied top-k/LOSO boundary. E3, E4 diagnostic slots, F1-B,
  and F1-C4 remain separate populations with no pooled funnel rate.
- The exporter uses Python standard-library modules only. The pre-format
  PONYTAIL audit reduced the initial 999-line staged version to 813 lines by
  removing unrequired multi-format adapters and speculative paths. The final
  981-line version adds only the PI-required single CSV and single TeX products.
- `/exports/paper_v2/*` is fixed to LF materialization so manifest-bound bytes
  remain deterministic on Windows and Linux.
- Stage Z appears only as `Stage Z pending` with the controlling
  `HOLD_STAGE_Z_Z0R2_OFT_CHECKPOINT_AUTHORITY_NOT_ESTABLISHED` status, no
  scientific rollout, and eight zero execution/protected counters. No Stage Z
  result is exported.
- The export manifest records Eval160 and protected evaluation as `UNREAD` and
  all new model/simulator/attack/physical/protected counters as zero.

## PONYTAIL and cleanup outcome

- No new dependency, plugin layer, template engine, renderer framework, or
  compatibility hierarchy was introduced.
- No authority file was moved or deleted. Historical top-level investigation
  scripts remain compatibility-preserved and explicitly noncanonical rather
  than being removed on filename appearance.
- The only top-level addition is a one-line, path-scoped `.gitattributes` rule
  needed for cross-platform deterministic export bytes.

## Terminal stop

`REPOSITORY_HYGIENE_PAPER_SUPPORT_READY_FOR_PI_REVIEW`

STOP. Do not merge automatically.
