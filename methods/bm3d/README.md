```bash
$ cd despeckling

$ mamba env create -f methods/bm3d/environment.yml

$ conda activate despeckling_bm3d

$ pip install -e .

$ bm3d \
    --input_path ../dataset/data_village.npz \
    --output_path methods/bm3d/results/output_village_bm3d.mat \
    --sigma_psd 15
```
