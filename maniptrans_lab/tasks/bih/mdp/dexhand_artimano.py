"""artiMANO constants copied from maniptrans_envs dexhands/artimano.py, kept as plain data for the lab venv."""

BODY_NAMES = [
    "palm",
    "index1y", "index1z", "index2", "index3", "index_tip",
    "middle1y", "middle1z", "middle2", "middle3", "middle_tip",
    "pinky1y", "pinky1z", "pinky2", "pinky3", "pinky_tip",
    "ring1y", "ring1z", "ring2", "ring3", "ring_tip",
    "thumb1x", "thumb1y", "thumb1z", "thumb2y", "thumb2z", "thumb3", "thumb_tip",
]

DOF_NAMES = [
    "j_index1y", "j_index1z", "j_index2", "j_index3",
    "j_middle1y", "j_middle1z", "j_middle2", "j_middle3",
    "j_pinky1y", "j_pinky1z", "j_pinky2", "j_pinky3",
    "j_ring1y", "j_ring1z", "j_ring2", "j_ring3",
    "j_thumb1x", "j_thumb1y", "j_thumb1z", "j_thumb2y", "j_thumb2z", "j_thumb3",
]

CONTACT_BODY_NAMES = ["thumb3", "index3", "middle3", "ring3", "pinky3"]

WEIGHT_IDX = {
    "thumb_tip": [27],
    "index_tip": [5],
    "middle_tip": [10],
    "ring_tip": [20],
    "pinky_tip": [15],
    "level_1_joints": [1, 2, 6, 7, 11, 12, 16, 17, 21, 22, 23],
    "level_2_joints": [3, 4, 8, 9, 13, 14, 18, 19, 24, 25, 26],
}

# PID wrist gains (artimano.py, PID-controlled wrist mode)
KP_ROT, KI_ROT, KD_ROT = 0.3, 0.01, 0.005
KP_POS, KI_POS, KD_POS = 10.0, 0.003, 0.5

N_BODIES = len(BODY_NAMES)
N_DOFS = len(DOF_NAMES)

# Isaac Gym-side drive and shape constants (dexhandmanip_bih.py)
DOF_STIFFNESS = 500.0
DOF_DAMPING = 30.0
HAND_FRICTION = 4.0
HAND_ROLLING_FRICTION = 0.01
HAND_TORSION_FRICTION = 0.01
