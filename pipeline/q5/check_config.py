"""CPU-only checks of real training samples and checkpoint metadata; never trains."""
import argparse
import dataclasses
import json
from pathlib import Path
from pipeline.common.reporting import emit_report

from pipeline.q5.runtime import Paths, prepare_environment, require_stats, require, validate_split, FREEZE_POLICY


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path)
    parser.add_argument("--output", type=Path, help="Optionally save JSON reports; otherwise print results only")
    args = parser.parse_args()
    paths = Paths.create(args.workspace)
    revision = prepare_environment(paths, cpu=True)
    stats_report = require_stats(paths)
    from pipeline.q5.config import build_config, require_base_assets
    require_base_assets(paths)
    import flax.nnx as nnx
    import numpy as np
    import pyarrow.parquet as pq
    import torch
    from openpi import transforms
    from openpi.training import data_loader
    torch.set_num_threads(2)
    config = build_config(paths)
    data = config.data.create(config.assets_dirs, config.model)
    require(data.use_quantile_norm and data.prompt_from_task, "Wrong Q5 normalization/prompt configuration")
    raw = data_loader.create_torch_dataset(data, config.model.action_horizon, config.model)
    base_dataset = raw
    while hasattr(base_dataset, "_dataset"):
        base_dataset = base_dataset._dataset
    require(base_dataset.video_backend == "pyav", "Q5 video backend must be pyav")
    dataset = data_loader.transform_dataset(raw, data)
    info = json.loads((paths.train / "meta/info.json").read_text())
    require(len(dataset) == info["total_frames"], "Unexpected training population")
    manifest = validate_split(paths)
    from pipeline.q5.data_contract import check_data_contract
    contract_report = check_data_contract(paths, data, manifest)
    starts = np.cumsum([0] + [r["length"] for r in manifest["train"][:-1]])
    tokenizer_transform = next(t for t in data.model_transforms.inputs if isinstance(t, transforms.TokenizePrompt))
    tokenizer = tokenizer_transform.tokenizer
    results = []
    maximum_tokens = 0
    for ep in [0, len(manifest["train"]) // 2, len(manifest["train"]) - 1]:
        n = manifest["train"][ep]["length"]
        for frame in [0, n // 2, n - 1]:
            index = int(starts[ep] + frame)
            item = dataset[index]
            require(item["state"].shape == (38,) and item["actions"].shape == (10, 38), "Incorrect transformed dimensions")
            require(np.isfinite(item["state"]).all() and np.isfinite(item["actions"]).all(), "Non-finite normalized sample")
            require(item["tokenized_prompt"].shape == (config.model.max_token_len,), "State token shape mismatch")
            require(item["image"]["base_0_rgb"].shape == (224, 224, 3), "Image shape mismatch")
            require(bool(item["image_mask"]["base_0_rgb"]) and not bool(item["image_mask"]["left_wrist_0_rgb"])
                    and not bool(item["image_mask"]["right_wrist_0_rgb"]), "Incorrect camera masks")
            repacked = transforms.compose([*data.repack_transforms.inputs, *data.data_transforms.inputs,
                                          transforms.Normalize(data.norm_stats, use_quantiles=True)])(raw[index])
            state_string = " ".join(map(str, np.digitize(repacked["state"], bins=np.linspace(-1, 1, 257)[:-1]) - 1))
            prompt = repacked["prompt"].strip().replace("_", " ").replace("\n", " ")
            token_count = len(tokenizer._tokenizer.encode(f"Task: {prompt}, State: {state_string};\nAction: ", add_bos=True))
            require(token_count <= config.model.max_token_len, "Real Q5 state tokens would be truncated")
            changed = repacked["state"].copy()
            changed[37] = -1.0 if changed[37] > 0 else 1.0
            original_tokens, _ = tokenizer.tokenize(prompt, repacked["state"])
            changed_tokens, _ = tokenizer.tokenize(prompt, changed)
            require(not np.array_equal(original_tokens, changed_tokens), "38th state joint did not affect tokens")
            maximum_tokens = max(maximum_tokens, token_count)
            results.append({"episode": ep, "frame": frame, "tokens": token_count})
    # Check token lengths for every training state without decoding all videos.
    maximum_sample_tokens = maximum_tokens
    tasks = {t["task_index"]: t["task"] for t in
             (json.loads(line) for line in (paths.train / "meta/tasks.jsonl").read_text().splitlines() if line.strip())}
    training_states_checked = 0
    state_normalizer = transforms.Normalize({"state": data.norm_stats["state"]}, use_quantiles=True)
    for record in manifest["train"]:
        table = pq.read_table(paths.train / record["data_path"], columns=["observation.state", "task_index"])
        states = np.asarray(table["observation.state"].to_pylist(), dtype=np.float32)
        normalized_states = state_normalizer({"state": states})["state"]
        codes = np.digitize(normalized_states, bins=np.linspace(-1, 1, 257)[:-1]) - 1
        for state_codes, task_id in zip(codes, table["task_index"].to_pylist(), strict=True):
            prompt = tasks[task_id].strip().replace("_", " ").replace("\n", " ")
            state_string = " ".join(map(str, state_codes))
            count = len(tokenizer._tokenizer.encode(f"Task: {prompt}, State: {state_string};\nAction: ", add_bos=True))
            require(count <= config.model.max_token_len, "A training state prompt exceeds the token limit")
            maximum_tokens = max(maximum_tokens, count)
            training_states_checked += 1
    require(training_states_checked == len(dataset), "Incomplete training-state token audit")
    # Validation is loaded separately, with the training statistics. No test
    # observations are read by either sample pipeline.
    validation_data = dataclasses.replace(data, repo_id="q5/pick_orange_to_basket_v2_validation")
    if hasattr(validation_data, "repo_root"):
        validation_data = dataclasses.replace(validation_data, repo_root=str(paths.validation))
    validation_raw = data_loader.create_torch_dataset(validation_data, config.model.action_horizon, config.model)
    validation_dataset = data_loader.transform_dataset(validation_raw, validation_data)
    validation_frames = sum(r["length"] for r in manifest["validation"])
    require(len(validation_dataset) == validation_frames, "Wrong validation population")
    validation_token_maximum = 0
    validation_states_checked = 0
    for record in manifest["validation"]:
        table = pq.read_table(paths.validation / record["data_path"], columns=["observation.state", "task_index"])
        states = np.asarray(table["observation.state"].to_pylist(), dtype=np.float32)
        normalized = state_normalizer({"state": states})["state"]
        codes = np.digitize(normalized, bins=np.linspace(-1, 1, 257)[:-1]) - 1
        for state_codes, task_id in zip(codes, table["task_index"].to_pylist(), strict=True):
            prompt = tasks[task_id].strip().replace("_", " ").replace("\n", " ")
            state_string = " ".join(map(str, state_codes))
            count = len(tokenizer._tokenizer.encode(f"Task: {prompt}, State: {state_string};\nAction: ", add_bos=True))
            require(count <= config.model.max_token_len, "A validation state prompt exceeds the token limit")
            validation_token_maximum = max(validation_token_maximum, count)
            validation_states_checked += 1
    require(validation_states_checked == validation_frames, "Incomplete validation-state token audit")
    for index in [0, validation_frames // 2, validation_frames - 1]:
        item = validation_dataset[index]
        require(item["state"].shape == (38,) and item["actions"].shape == (10, 38), "Invalid validation sample")
        require(np.isfinite(item["state"]).all() and np.isfinite(item["actions"]).all(), "Invalid normalized validation data")
    predicate = nnx.filterlib.to_predicate(config.trainable_filter)
    variable = nnx.Param(np.zeros((1,), dtype=np.float32))
    component_trainability = {}
    for path, trainable in [(('action_in_proj', 'kernel'), True), (('action_out_proj', 'bias'), True),
                            (('PaliGemma', 'llm', 'test_lora', 'a'), True),
                            (('PaliGemma', 'llm', 'kernel'), False), (('PaliGemma', 'img', 'kernel'), True),
                            (('time_mlp_in', 'kernel'), True), (('time_mlp_out', 'kernel'), True)]:
        require(predicate(path, variable) == trainable, f"Unexpected freeze rule: {path}")
        component_trainability["/".join(path)] = bool(predicate(path, variable))
    # Exercise the official batched input pipeline, without model initialization.
    loader = data_loader.create_data_loader(config, shuffle=False, num_batches=1)
    observation, actions = next(iter(loader))
    require(tuple(observation.state.shape) == (1, 38) and tuple(actions.shape) == (1, 10, 38), "Incorrect real batch")
    from pipeline.q5.parameter_audit import audit_checkpoint_metadata
    metadata_audit = audit_checkpoint_metadata(paths.base_params / "_METADATA", config.trainable_filter)
    report = dict(status="PASS", upstream_commit=revision, train_episodes=len(manifest["train"]),
                  validation_episodes=len(manifest["validation"]), test_episodes=10, validation_frames=validation_frames,
                  training_frames=len(dataset), test_episodes_in_training=0, samples=results,
                  maximum_checked_tokens=maximum_tokens, token_limit=config.model.max_token_len,
                  maximum_sample_tokens=maximum_sample_tokens, training_state_token_frames_checked=training_states_checked,
                  validation_state_token_frames_checked=validation_states_checked, validation_maximum_tokens=validation_token_maximum,
                  batch_state_shape=[1, 38], batch_action_shape=[1, 10, 38],
                  norm_stats_sha256=stats_report["sha256"], trainable_rule_checked=True,
                  checkpoint_metadata_checked=True, cpu_only=True, video_backend=base_dataset.video_backend,
                  freeze_policy=FREEZE_POLICY, trainable_components=component_trainability,
                  finetune_learning_rate=dataclasses.asdict(build_config(paths, profile="finetune").lr_schedule),
                  finetune_initial_steps=build_config(paths, profile="finetune").num_train_steps,
                  data_contract=contract_report,
                  real_base_metadata_parameter_audit={k: v for k, v in metadata_audit.items() if k != "parameters"},
                  full_weight_loading_tested=False, model_forward_tested=False, training_tested=False)
    if args.output is not None:
        output = args.output.resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.with_name(output.stem + "_base_parameter_metadata.json").write_text(json.dumps(metadata_audit, indent=2) + "\n")
    emit_report(report, args.output)


if __name__ == "__main__":
    main()
