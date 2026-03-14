#!/bin/bash
# =============================================================================
# ManipTrans Stage 2: Residual policy training (object manipulation)
#
# Trains a residual policy on top of a frozen stage 1 (hand imitation) base.
# The base model architecture is auto-detected from its checkpoint.
#
# Usage:
#   bash maniptrans_residual.sh GPU_ID [EXP_NAME] [residual_action_scale] [learning_rate] [lr_schedule] [sigma_init_val] [tighten_steps]
#
# Examples:
#   bash maniptrans_residual.sh 0                               # defaults
#   bash maniptrans_residual.sh 0 my_exp                        # custom name
#   bash maniptrans_residual.sh 0 abl_lr 2.0 2e-4              # scale + lr
#   bash maniptrans_residual.sh 0 abl_all 2.0 2e-4 warmup -1 3200
# =============================================================================

PROJECT_DIR="/home/nas4_dataset/human"
cd "${PROJECT_DIR}/Code/ManipTrans"

source /home/nas4_user/sibeenkim/anaconda3/etc/profile.d/conda.sh
conda activate maniptrans

# --- Fixed params ---
DEXHAND="xhand"
SIDE="RH"
DATA_DIR="${PROJECT_DIR}/handphuma/${DEXHAND}"
OBJ_DIR="${PROJECT_DIR}/handphuma/object"
DATA_TYPE="grab_demo"
EXP_NAME="smoke_res_scale"

# --- Environment ---
num_envs=4096
max_iterations=10000000
horizon_length=32
num_mini_batches=4
minibatch_size=$(( num_envs * horizon_length / num_mini_batches ))

# --- PhysX buffers (object collision geometry needs larger buffers) ---
contact_pairs_multiplier=4      # default: 1 (applied to base 1048576; vec_task.py runtime doubles again)
buffer_size_multiplier=8        # default: 5 (scales foundLostAggregatePairsCapacity etc.)

# --- Training params (defaults; override via positional args) ---
learning_rate=2e-4
lr_schedule=warmup
grad_norm=1.0
entropy_coef=0.0
kl_threshold=0.008
tighten_steps=256000
tighten_factor=0.7
sampling_method=twolevel
mastery_floor=0.0
sonic_blend_alpha=0.0
twolevel_chunk_alpha=0.0
bin_ema_alpha=1.0
chunk_metrics_log_every=1
eval_every=200
actions_moving_average=0.4
sigma_init_val=0
residual_action_scale=2.0

# --- Parse arguments ---
# $1=GPU_ID  $2=EXP_NAME  $3=residual_action_scale  $4=learning_rate
# $5=lr_schedule  $6=sigma_init_val  $7=tighten_steps
GPU_ID=${1:-0}
EXP_NAME=${2:-${EXP_NAME}}
residual_action_scale=${3:-${residual_action_scale}}
learning_rate=${4:-${learning_rate}}
lr_schedule=${5:-${lr_schedule}}
sigma_init_val=${6:-${sigma_init_val}}
tighten_steps=${7:-${tighten_steps}}
# $8=no_bps_obj_meta: if "1", disable BPS + obj_com + obj_weight (reduces privilegedObsDim by 4)
no_bps_obj_meta=${8:-0}
# $9=finger_tip_force_weight: reward weight for finger tip force (default 1.0)
finger_tip_force_weight=${9:-1.0}
BASE_CHECKPOINT="runs/alldata_fixedlr_separate_ema04_v1_xhand__03-09-01-38-30/nn/epoch_600.pth"
EXTRA_ARGS=()

if [ "${no_bps_obj_meta}" = "1" ]; then
    # ndof=12: privileged = 12+13+20 = 45 (removed obj_com=3, obj_weight=1)
    EXTRA_ARGS+=(useBPS=false)
    EXTRA_ARGS+=("task.env.privilegedObsKeys=[dq,manip_obj_pos,manip_obj_quat,manip_obj_vel,manip_obj_ang_vel,tip_force]")
    EXTRA_ARGS+=(task.env.privilegedObsDim=45)
    EXTRA_ARGS+=("rl_train.params.network.dict_feature_encoder.extractors.privileged.input_dim=45")
fi

if [ ! -f "${BASE_CHECKPOINT}" ]; then
    echo "ERROR: Base checkpoint not found: ${BASE_CHECKPOINT}"
    exit 1
fi

# --- Build index path ---
INDEX_PARTS=""
IFS=',' read -ra SPLITS <<< "${DATA_TYPE}"
for split in "${SPLITS[@]}"; do
    split=$(echo "$split" | xargs)
    if [ ! -f "${DATA_DIR}/${split}.pt" ]; then
        echo "ERROR: ${DATA_DIR}/${split}.pt not found"
        exit 1
    fi
    if [ -n "${INDEX_PARTS}" ]; then
        INDEX_PARTS="${INDEX_PARTS},${split}.pt"
    else
        INDEX_PARTS="${split}.pt"
    fi
    echo "  Using: ${DATA_DIR}/${split}.pt"
done
INDEX_PATH="${INDEX_PARTS}"

export CUDA_VISIBLE_DEVICES="${GPU_ID}"

echo "============================================"
echo "  Stage 2 Residual Training"
echo "  GPU=${GPU_ID}  Dataset: ${DATA_TYPE}  Experiment: ${EXP_NAME}"
echo "  Base (stage1): ${BASE_CHECKPOINT}"
echo "  num_envs=${num_envs}"
echo "  minibatch_size=${minibatch_size}  (auto: ${num_envs}*${horizon_length}/${num_mini_batches})"
echo "  learning_rate=${learning_rate}  lr_schedule=${lr_schedule}"
echo "  grad_norm=${grad_norm}, entropy_coef=${entropy_coef}, kl_threshold=${kl_threshold}"
echo "  tighten_steps=${tighten_steps}, tighten_factor=${tighten_factor}"
echo "  actions_moving_average=${actions_moving_average}"
echo "  residual_action_scale=${residual_action_scale}"
echo "  contact_pairs_multiplier=${contact_pairs_multiplier} (effective: $((1048576 * contact_pairs_multiplier * 2)))"
echo "  buffer_size_multiplier=${buffer_size_multiplier}"
echo "  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "============================================"

python main/rl/train.py \
    task=ResDexHand \
    dexhand=${DEXHAND} \
    side=${SIDE} \
    headless=true \
    num_envs=${num_envs} \
    minibatch_size=${minibatch_size} \
    usePIDControl=True \
    randomStateInit=true \
    actionsMovingAverage=${actions_moving_average} \
    learning_rate=${learning_rate} \
    max_iterations=${max_iterations} \
    tighten_steps=${tighten_steps} \
    tighten_factor=${tighten_factor} \
    sampling_method=${sampling_method} \
    mastery_floor=${mastery_floor} \
    sonic_blend_alpha=${sonic_blend_alpha} \
    twolevel_chunk_alpha=${twolevel_chunk_alpha} \
    bin_ema_alpha=${bin_ema_alpha} \
    chunk_metrics_log_every=${chunk_metrics_log_every} \
    eval_every=${eval_every} \
    rl_train.params.config.lr_schedule=${lr_schedule} \
    rl_train.params.config.entropy_coef=${entropy_coef} \
    rl_train.params.config.grad_norm=${grad_norm} \
    rl_train.params.config.kl_threshold=${kl_threshold} \
    rl_train.params.network.space.continuous.sigma_init.val=${sigma_init_val} \
    rh_base_model_checkpoint=${BASE_CHECKPOINT} \
    experiment=${EXP_NAME} \
    data_dir=${DATA_DIR} \
    obj_dir=${OBJ_DIR} \
    task.sim.physx.max_gpu_contact_pairs=$((1048576 * contact_pairs_multiplier)) \
    task.sim.physx.default_buffer_size_multiplier=${buffer_size_multiplier} \
    "index_path='${INDEX_PATH}'" \
    task.env.residualActionScale=${residual_action_scale} \
    +finger_tip_force_weight=${finger_tip_force_weight} \
    "${EXTRA_ARGS[@]}"
