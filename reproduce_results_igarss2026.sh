#!/bin/bash

# ~/miniforge3/envs/despeckling_gbf/bin/python ./baselines/gbf/run.py \
#     --input_path_sar ./data/data.npz \
#     --input_path_opt ./data/data.npz \
#     --matlab_script_path ./../SARBM3D_v10_linux64/SARBM3D_v10.m \
#     --output_path ./baselines/gbf/results/output.mat \
#     --L 1 \
#     --window_size 15 \
#     --lambda_S 0.005 \
#     --lambda_RO 0.02 \
#     --lambda_RS 0.1 \
#     --gamma 7 \
#     --a0 0.64 \
    # --N "7 9 11 13 15 17 19 21 23 25 27 29 31 33 35 37 39 41 43 45 47 49 51 53 55 57 59 61"

# ~/miniforge3/envs/despeckling_gnlm/bin/python ./baselines/gnlm/run.py \
#     --input_path_sar ./data/data.npz \
#     --input_path_opt ./data/data.npz \
#     --matlab_script_path ./../GNLM/matlab/guidedNLMeans.m \
#     --output_path ./baselines/gnlm/results/output.mat \
#     --L 1 \
#     --stack_size 256 \
#     --sharpness 0.002 \
#     --balance 0.15 \
#     --th_sar 2.0 \
#     --block_size 8 \
#     --win_size 39 \
#     --stride 3

# ~/miniforge3/envs/despeckling_aef/bin/python ./baselines/aef/run.py \
#     --input_path_sar ./data/data.npz \
#     --input_path_embeddings ./data/data.npz \
#     --output_path ./baselines/aef/results/output.mat \
#     --sigma_distance 0.1 \
#     --radius_despeckling 30 \
#     --n_blocks 10 

~/miniforge3/envs/despeckling_tbog/bin/python ./scripts/run.py \
    --input_path_sar ./data/data.npz \
    --input_path_opt ./data/data.npz \
    --output_path ./results/output.mat \
    --radius_descriptor 7 \
    --sigma_spatial 5 \
    --sigma_guides 0.01 0.01 \
    --gamma_guides 1.0 1.0 \
    --alpha 1 \
    --n_iterations 100 \
    --sigma_distance 1.5 \
    --radius_despeckling 30 \
    --n_blocks 20
