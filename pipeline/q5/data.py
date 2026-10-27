"""One head camera, complete 38-dimensional states and absolute joint targets."""
import dataclasses
import numpy as np

from openpi import transforms
from openpi.training import config as upstream


@dataclasses.dataclass(frozen=True)
class Q5Inputs:
    def __call__(self, data):
        state = np.asarray(data["state"], dtype=np.float32)
        if state.shape != (38,) or not np.isfinite(state).all():
            raise ValueError("Q5 state must contain 38 finite joint values")
        image = np.asarray(data["images"]["image_head"])
        if image.ndim == 3 and image.shape[0] == 3 and image.shape[-1] != 3:
            image = np.moveaxis(image, 0, -1)
        if image.ndim != 3 or image.shape[-1] != 3 or not np.isfinite(image).all():
            raise ValueError("Invalid Q5 RGB image")
        if np.issubdtype(image.dtype, np.floating):
            if image.min() < 0 or image.max() > 1:
                raise ValueError("Floating-point Q5 images must be in [0, 1]")
            image = np.rint(image * 255).astype(np.uint8)
        elif image.dtype != np.uint8:
            raise ValueError("Integer Q5 images must be uint8")
        result = {"state": state, "image": {"base_0_rgb": image,
                  "left_wrist_0_rgb": np.zeros_like(image), "right_wrist_0_rgb": np.zeros_like(image)},
                  "image_mask": {"base_0_rgb": np.True_, "left_wrist_0_rgb": np.False_, "right_wrist_0_rgb": np.False_}}
        if "actions" in data:
            actions = np.asarray(data["actions"], dtype=np.float32)
            if actions.ndim != 2 or actions.shape[-1] != 38 or not np.isfinite(actions).all():
                raise ValueError("Q5 actions must be a finite horizon-by-38 array")
            result["actions"] = actions
        if "prompt" in data:
            result["prompt"] = data["prompt"]
        return result


@dataclasses.dataclass(frozen=True)
class Q5Outputs:
    def __call__(self, data):
        actions = np.asarray(data["actions"])
        if actions.ndim != 2 or actions.shape[-1] != 38 or not np.isfinite(actions).all():
            raise ValueError("Q5 policy output must be finite horizon-by-38")
        return {"actions": actions}


@dataclasses.dataclass(frozen=True)
class Q5DataConfig(upstream.DataConfigFactory):
    repo_root: str | None = None

    def create(self, assets_dirs, model_config):
        changes = dict(prompt_from_task=True, action_sequence_keys=("action",),
                       repack_transforms=transforms.Group(inputs=[transforms.RepackTransform({
                           "state": "observation.state", "images": {"image_head": "observation.images.image_head"},
                           "actions": "action", "prompt": "prompt"})]),
                       data_transforms=transforms.Group(inputs=[Q5Inputs()], outputs=[Q5Outputs()]),
                       model_transforms=upstream.ModelTransformFactory()(model_config))
        if "repo_root" in {f.name for f in dataclasses.fields(upstream.DataConfig)}:
            changes["repo_root"] = self.repo_root
        return dataclasses.replace(self.create_base_config(assets_dirs, model_config), **changes)
