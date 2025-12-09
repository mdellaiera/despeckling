```bash
$ cd despeckling

$ mamba env create -f methods/bm3d/environment.yml

$ conda activate despeckling_bm3d

$ pip install -e .

$ bm3d \
    --input_path ../dataset/data.npz \
    --sigma_psd 10
```
