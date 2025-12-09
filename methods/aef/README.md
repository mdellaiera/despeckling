```bash
$ cd despeckling

$ mamba env create -f methods/aef/environment.yml

$ conda activate despeckling_aef

$ pip install -e .

$ aef \
    --input_path ../dataset/data_changi.npz \
    --sigma_distance 0.1 \
    --radius_despeckling 30 \
    --n_blocks 20 
```
