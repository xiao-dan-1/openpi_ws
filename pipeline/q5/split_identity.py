"""Verify a relocated split against the manifest recorded during training."""
import json
from pathlib import Path

from pipeline.q5.runtime import require, sha256


def verify_training_split(local_path, expected_sha256, reference_path=None):
    local_path = Path(local_path)
    local_sha256 = sha256(local_path)
    if reference_path is None:
        require(local_sha256 == expected_sha256,
                "Checkpoint split differs; supply --training-split-manifest with the original training manifest")
        return {"comparison": "exact_manifest_sha256", "training_sha256": expected_sha256,
                "local_sha256": local_sha256}

    reference_path = Path(reference_path)
    require(reference_path.is_file(), f"Training split manifest not found: {reference_path}")
    require(sha256(reference_path) == expected_sha256,
            "Training split manifest SHA256 differs from the training record")
    local = json.loads(local_path.read_text())
    reference = json.loads(reference_path.read_text())
    for role in ["train", "validation", "test"]:
        for actual, recorded in zip(local[role], reference[role]):
            a, b = actual.get("original_episode_index"), recorded.get("original_episode_index")
            require(a is None or b is None or a == b,
                    f"Conflicting original episode identities in {role}")

    def identity(manifest):
        # Machine paths and optional pre-cleaning history do not define training samples.
        result = {k: v for k, v in manifest.items() if k not in {
            "source", "cleaning_history_available", "cleaning_manifest_sha256", "episode_identity"}}
        for role in ["train", "validation", "test"]:
            result[role] = [{k: v for k, v in row.items() if k != "original_episode_index"}
                            for row in manifest[role]]
        return json.dumps(result, sort_keys=True, separators=(",", ":"))

    require(identity(local) == identity(reference),
            "Checkpoint split differs in episode identities, data fingerprints or split settings")
    return {"comparison": "verified_data_identity", "training_sha256": expected_sha256,
            "local_sha256": local_sha256, "training_manifest": str(reference_path.resolve())}
