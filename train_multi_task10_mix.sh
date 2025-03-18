TASK=10tasks_selected_keyframe_state
FUTURE_ACTION_STEPS=15
skip_layer_number=$1
mse_weight=$2
balance_weight=$3
KD_weight=$4
add_lstm=False
route_in_for=False
use_index=False # 为False是就是目前的sota strategy
seed=$5
ema_decay=$6
# SETTING=freeze_vit_window${FUTURE_ACTION_STEPS}
SETTING=route_in_for_${route_in_for}_balance_${balance_weight}_KD_${KD_weight}_ema_${ema_decay}_seed_${seed}_skip_layer_${skip_layer_number}_mse_${mse_weight}_freeze_none_window${FUTURE_ACTION_STEPS}
# SETTING=index_strategy_freeze_none_window${FUTURE_ACTION_STEPS}_Llama2_MoE_bz_4_new_DiT_skip_llama_layer_${skip_layer_number}_lstm_architecture_${add_lstm}_${TASK}_route_in_for_${route_in_for}
# SETTING=freeze_none_window${FUTURE_ACTION_STEPS}_Llama2_MoE_bz_4_new_DiT_baseline_test
echo ${SETTING}
# SETTING=freeze_none_window${FUTURE_ACTION_STEPS}_Llama2_bz_4_vla_llm_align_baseline_task3
FREEZE_VISON=true
FREEZE_LLM=False
LOAD_DIT=true
# LLM_LAYER=mix2_AdaptiveLayerDefaultDit
# LLM_LAYER=mix_avgNoneDpPooling_m2Pooling
LLM_LAYER=mix_freezeLLM_AdaptiveLayerDefaultDitsWithLearnable_test
export HF_HOME=/home/huggingface


# device='0,1,2,3,4,5,6,7'  # 一定要设这个才能使用FDSP！！！
# device='4,5,6,7'  # 一定要设这个才能使用FDSP！！！
device=$7  # 一定要设这个才能使用FDSP！！！
# device='4'  # 一定要设这个才能使用FDSP！！！

current_dir=$(pwd)
# 提取路径的第二部分并构造 ckpt_root
second_dir=$(echo "$current_dir" | cut -d"/" -f2)
export PATH=$PATH:/$second_dir/dmh/Cogact_test
export PYTHONPATH=$PYTHONPATH:/$second_dir/dmh/Cogact_test  # CogAct official model
export PATH=$PATH:/$second_dir/dmh/Cogact_test/vla
export PYTHONPATH=$PYTHONPATH:/$second_dir/dmh/Cogact_test/vla 

export EMA_DECAY=${ema_decay}
export MSE=${mse_weight}
export BALANCE=${balance_weight}
export KD=${KD_weight}

SETTING=${SETTING} COG_RES=${cog_res} RANDOM_SEED=${seed} USE_INDEX=${use_index} FOR_ROUTE=${route_in_for} TRAIN_ROUTE=False ADD_LSTM=${add_lstm} SKIP_LAYER_NUMBER=$skip_layer_number CUDA_VISIBLE_DEVICES=$device torchrun --standalone --nnodes 1 --nproc-per-node 8 scripts/train.py \
  --vla.type prism-dinosiglip-224px+oxe+diffusion \
  --vla.data_mix rlbench \
  --vla.expected_world_size 8 \
  --vla.global_batch_size 512 \
  --vla.per_device_batch_size 64 \
  --vla.learning_rate 2e-5 \
  --vla.epochs 100 \
  --vla.freeze_vision_backbone ${FREEZE_VISON} \
  --vla.freeze_llm_backbone ${FREEZE_LLM} \
  --data_root_dir /home/dmh/hydra/cogact/datasets/${TASK} \
  --run_root_dir /home/dmh/CogACT_test/checkpoint/${TASK}_cogact_large \
  --run_id exp_cx_LLMLAYER_${LLM_LAYER}_${TASK}_${SETTING} \
  --image_aug false \
  --save_interval 800 \
  --action_dim 7 \
  --repeated_diffusion_steps 8 \
  --future_action_window_size ${FUTURE_ACTION_STEPS} \
  --load_dit ${LOAD_DIT} \
  --action_model_type DiT-B \
  --is_resume False \
  --pretrained_checkpoint "/home/cx/chenhao/hub/models--CogACT--CogACT-Base/snapshots/ffc4db3bef7735ba7aa692d50b6454588a32b753/checkpoints/CogACT-Base.pt" \





# tasks=("put_rubbish_in_bin_sparse" "toilet_seat_down_sparse" "unplug_charger_sparse" "close_laptop_lid_sparse" "water_plants_sparse")

# for task in "${tasks[@]}"; do
#   echo "Running training for task: $task"
  
#   torchrun --standalone --nnodes 1 --nproc-per-node 8 scripts/train.py \
#     --pretrained_checkpoint "/home/cx/chenhao/hub/models--CogACT--CogACT-Base/snapshots/ffc4db3bef7735ba7aa692d50b6454588a32b753/checkpoints/CogACT-Base.pt" \
#     --vla.type prism-dinosiglip-224px+oxe+diffusion \
#     --vla.data_mix rlbench \
#     --vla.expected_world_size 8 \
#     --vla.global_batch_size 256 \
#     --vla.per_device_batch_size 32 \
#     --vla.learning_rate 2e-5 \
#     --data_root_dir /home/cx/rlds_dataset_builder/dataset/${task} \
#     --run_root_dir /home/cx/4dvla/CogACT \
#     --run_id exp_${task}_freeze_vit_window15 \
#     --image_aug false \
#     --wandb_project cogact \
#     --wandb_entity 1162737898-the-chinese-university-of-hong-kong \
#     --save_interval 100 \
#     --repeated_diffusion_steps 8 \
#     --future_action_window_size 15 \
#     --action_model_type DiT-B \
#     --is_resume False
  
#   echo "Finished training for task: $task"
# done





#### close_box  close_laptop_lid   put_rubbish_in_bin  unplug_charger  water_plants  toilet_seat_down