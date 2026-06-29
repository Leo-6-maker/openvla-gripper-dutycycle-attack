# Server Runtime Reconciliation V1

Status: `SERVER_SNAPSHOT_REQUIRED`

No Bubble/local server snapshot has been received in this PR revision. The table below records the required comparison surface and the previously observed live-audit SHA where available. Bubble is a transport layer only; these rows remain unresolved until a small snapshot with `SHA256SUMS.txt` and metadata is verified locally.

| Server path | Server SHA256 | GitHub path | GitHub SHA256 | Byte-identical | Semantic difference | Used by running CLEAN | Acceptable for future Batch A | Required action |
|---|---:|---|---:|---|---|---|---|---|
| `/mnt/sdc/dty_user/openvla_attack/scripts/stageb/run_vis_formal_worker.py` | `a7bdd8601a32fe835ced3af681d036861ba246c19ad2edcfea5797749890e0928` | `scripts/stageb/run_vis_formal_worker.py` | `SERVER_SNAPSHOT_REQUIRED` | `SERVER_SNAPSHOT_REQUIRED` | `SERVER_SNAPSHOT_REQUIRED` | yes | unresolved | Bubble snapshot and file diff required |
| `/mnt/sdc/dty_user/openvla_attack/scripts/stageb/run_v2_vis_sc5_mlp_bridge.py` | `cf125d2393f2ca0a5ec1b62610b22b8d5c17733a647b9824edd2aab19995daa6` | `scripts/stageb/run_v2_vis_sc5_mlp_bridge.py` | `SERVER_SNAPSHOT_REQUIRED` | `SERVER_SNAPSHOT_REQUIRED` | `SERVER_SNAPSHOT_REQUIRED` | yes | unresolved | Bubble snapshot and semantic diff required |
| `/mnt/sdc/dty_user/openvla_attack/configs/cross_suite_clean1500_protocol_v1.json` | `449da97b2a0bcb339b19b629e19ffbab7590bf1bed0c4410b8fb114861633ce0` | `configs/cross_suite_clean1500_protocol_v1.json` | `SERVER_SNAPSHOT_REQUIRED` | `SERVER_SNAPSHOT_REQUIRED` | `SERVER_SNAPSHOT_REQUIRED` | CLEAN1500 | unresolved | Bubble snapshot and protocol diff required |
| `/mnt/sdc/dty_user/openvla_attack/configs/cross_suite_clean1500_protocol_v1_gpu7_goal.json` | `bb6f4033a80fef6e6884aac2013880409d7bfd777731a87c9412f13a9609dc99e` | `configs/cross_suite_clean1500_protocol_v1_gpu7_goal.json` | `SERVER_SNAPSHOT_REQUIRED` | `SERVER_SNAPSHOT_REQUIRED` | `SERVER_SNAPSHOT_REQUIRED` | CLEAN1500 GPU7 | unresolved | Bubble snapshot and accepted-variant note required |
| `/mnt/sdc/dty_user/openvla_attack/configs/cross_suite_object_target_registry_v1.json` | `b0c0ec6fb33dbc2b066f9c759b1762d5a2ebada7cce2f9559a99150e0a9ec750` | `configs/cross_suite_object_target_registry_v1.json` | `SERVER_SNAPSHOT_REQUIRED` | `SERVER_SNAPSHOT_REQUIRED` | `SERVER_SNAPSHOT_REQUIRED` | CLEAN1500 | unresolved | Bubble snapshot and registry diff required |
| `/mnt/sdc/dty_user/openvla_attack/evidence/sc5_object_privileged_loto_v1/vis_heldout_formal_v1/CLEAN/MANIFEST.jsonl` | `105f25a3b4bf1681eb4959ce63e17dec62296c0abb12db8ade969773ab375bc4` | none committed | n/a | n/a | live artifact, not GitHub source | yes | only after freeze | include manifest SHA in freeze bundle; do not commit rollout tree |

Required server snapshot location should stay under a dty-owned data directory such as `/mnt/sdc/dty_user/...`; do not use `/`, `/tmp`, or large live evidence-tree copies.
