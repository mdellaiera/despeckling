```bash
$ cd despeckling

$ mamba env create -f methods/tbog/environment.yml

$ conda activate despeckling_tbog

$ pip install -e .

$ tbog \
    --input_path ../dataset/data.npz \
    --radius_descriptor 7 \
    --sigma_spatial 5 \
    --sigma_luminance_eo 0.1 \
    --sigma_luminance_sar 0.1 \
    --gamma_luminance_eo 1.0 \
    --gamma_luminance_sar 1.0 \
    --alpha 1 \
    --n_iterations 100 \
    --sigma_distance 1.5 \
    --radius_despeckling 30 \
    --n_blocks 30
```
