STEP_FIELDS = [
    "version","run_id","experiment_id","task_id","suite","episode_id","seed","step_idx","max_steps",
    "trigger_name","rho","trigger_timing","raw_trigger_score","trigger_active_raw","trigger_request_active","attack_active",
    "budget_max_steps","budget_used_before","budget_remaining_after","budget_blocked","signal_available",
    "fallback","fallback_reason","oracle","privileged","Ntrig_fwd","Ntrig_decode","Nattack_fwd",
    "Nattack_decode","Tclean_decode","Ttrig","Tattack","Ttotal","Mcache","Asignal","clean_action",
    "executed_action","action_delta","action_bound_low","action_bound_high","nad_cleanref_dims",
    "nad_cleanref_step","directional_alignment","directional_alignment_cos","delta_l2",
    "delta_linf","attack_method","directional_loss_available","epsilon","attack_steps",
    "observation_perturb_linf","observation_perturb_l2",
    "target_token_ids","clean_token_ids","adv_token_ids","target_ce_initial","target_ce_final",
    "token_match_rate","token_changed_count","pixel_space","num_loss_forwards","num_backwards",
    "base_instruction","effective_instruction","instruction_override","instruction_suffix",
    "prompt_variants_path","offline_prompt_audit",
    "prompt_id","prompt_type","prompt_attack_config","target_primitive",
    "clean_gripper_token","attack_gripper_token","gripper_token_flip","target_primitive_ok",
    "success_so_far",
]
EPISODE_FIELDS = [
    "version","run_id","experiment_id","task_id","suite","seed","episode_id","trigger_name","rho",
    "success","failure","timeout","invalid","invalid_reason","num_steps","num_attack_active_steps",
    "attacked_step_ratio","raw_trigger_rate","trigger_request_rate","budget_blocked_rate",
    "action_delta_l2_all","action_delta_l2_attacked","action_delta_linf_all","action_delta_linf_attacked",
    "nad_cleanref_all","nad_cleanref_attacked",
    "mean_alignment_all","mean_alignment_attacked","targeted_alignment_rate","latency_total_p50",
    "latency_total_p95","latency_trigger_p50","latency_trigger_p95","latency_attack_p50",
    "latency_attack_p95","signal_availability_rate","fallback_rate","feasibility_pass","artifact_step_jsonl",
]
RUN_FIELDS = [
    "version","run_id","created_at","host","user","cwd","command","code_git_commit","code_dirty",
    "config_hash","attack_config_path","tasks_config_path","directions_config_path","thresholds_path",
    "model_id","model_checkpoint_path","dataset_manifest_hash","task_id","suite","seed","trigger_name",
    "rho","episodes","max_steps","output_files","status","error",
    "cuda_visible_devices","render_gpu_device_id","model_gpu_device_id",
    "instruction_override","instruction_suffix","prompt_variants_path","offline_prompt_audit",
    "prompt_id","prompt_type","prompt_attack_config","target_primitive",
    "matched_to_run_id","matched_to_trigger","matched_to_attacked_ratio",
]


def _validate(record: dict, fields: list[str], kind: str) -> None:
    missing = [f for f in fields if f not in record]
    if missing:
        raise ValueError(f"{kind} record missing fields: {missing}")


def validate_step_record(record: dict) -> None:
    _validate(record, STEP_FIELDS, "step")


def validate_episode_record(record: dict) -> None:
    _validate(record, EPISODE_FIELDS, "episode")


def validate_run_manifest(record: dict) -> None:
    _validate(record, RUN_FIELDS, "run_manifest")
