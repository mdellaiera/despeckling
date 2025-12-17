```bash
$ cd despeckling

$ mamba env create -f methods/tbog/environment.yml

$ conda activate despeckling_tbog

$ pip install -e .

$ tbog \
    --input_path ../dataset/data_village.npz \
    --output_path methods/tbog/results/output_village_tbog.mat \
    --radius_descriptor 7 \
    --sigma_spatial 5 \
    --sigma_guides 0.1 0.001 \
    --gamma_guides 1.0 1.0 \
    --alpha 5 \
    --n_iterations 100 \
    --sigma_distance 5 \
    --radius_despeckling 50 \
    --n_blocks 20

$ tbog \
    --input_path ../dataset/data_changi.npz \
    --output_path methods/tbog/results/output_changi_tbog.mat \
    --radius_descriptor 7 \
    --sigma_spatial 5 \
    --sigma_guides 0.1 0.01 \
    --gamma_guides 1.0 1.0 \
    --alpha 5 \
    --n_iterations 100 \
    --sigma_distance 1.5 \
    --radius_despeckling 50 \
    --n_blocks 20
```
