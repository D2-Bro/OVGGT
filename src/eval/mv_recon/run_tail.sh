#!/bin/bash
set -e

workdir='..'
model_name='OVGGT'
ckpt_name='checkpoints'
model_weights="ckpt/${ckpt_name}.pth"
max_frames='500'
seven_scenes_root="/data2/dongjae/datasets/7scenes_sfm"
seven_scenes_seq_id="seq-01"
export MV_RECON_KNN_QUERY_CHUNK=49152
export MV_RECON_KNN_REF_CHUNK=131072

# base_output_dir="eval_results/mv_recon/static_order/${model_name}"
base_output_dir="eval_results/mv_recon/ordering_exp_500_tail/${model_name}"

for i in $(seq 0 0); do
    run_id=$(printf "run_%03d" "$i")
    output_dir="${base_output_dir}/${run_id}"

    echo "Running ${run_id} -> ${output_dir}"

    accelerate launch --num_processes 4 --main_process_port $((29602 + i % 20)) src/eval/mv_recon/launch.py \
        --weights "$model_weights" \
        --output_dir "$output_dir" \
        --model_name "$model_name" \
        --max_frames "$max_frames" \
        --seven_scenes_root "$seven_scenes_root" \
        --seven_scenes_seq_id "$seven_scenes_seq_id" \
        --seed "$i" \
        --skip_save_artifacts 
done
