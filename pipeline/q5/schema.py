"""Q5 joint names in the existing robot contract order (not a control policy)."""
ARM_NAMES = tuple(f"{side}_{joint}_joint" for side in ("left", "right") for joint in (
    "shoulder_pitch", "shoulder_roll", "arm_yaw", "elbow_pitch", "elbow_yaw", "wrist_pitch", "wrist_roll"))
HAND_NAMES = tuple(f"{side}_hand_{joint}" for side in ("left", "right") for joint in (
    "thumb_bend_joint", "thumb_rota_joint1", "thumb_rota_joint2", "index_bend_joint", "index_joint1",
    "index_joint2", "mid_joint1", "mid_joint2", "ring_joint1", "ring_joint2", "pinky_joint1", "pinky_joint2"))
JOINT_NAMES = ARM_NAMES + HAND_NAMES
STATE_UNITS = "rad"
ACTION_UNITS = "rad"
ACTION_REPRESENTATION = "absolute_joint_targets"
