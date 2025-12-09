```bash
$ cd despeckling

$ mamba env create -f methods/gbf/environment.yml

$ conda activate despeckling_gbf

$ pip install -e .

$ gbf \
    --input_path_sar ../dataset/data.npz \
    --matlab_script_path .../SARBM3D_v10_linux64/SARBM3D_v10.m
    --L 1 \
    --window_size 15 \
    --lambda_S 0.005 \
    --lambda_RO 0.02 \
    --lambda_RS 0.1 \
    --gamma 7 \
    --a0 0.64
```
