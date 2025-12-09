```bash
$ ~/miniforge3/envs/despeckling_gbf/bin/python run.py \
    --input_path_sar PATH_TO_SAR_DATA \
    --input_path_opt PATH_TO_EO_DATA \
    --matlab_script_path ../../../SARBM3D_v10_linux64/SARBM3D_v10.m
    --L 1 \
    --window_size 15 \
    --lambda_S 0.005 \
    --lambda_RO 0.02 \
    --lambda_RS 0.1 \
    --gamma 7 \
    --a0 0.64
```
