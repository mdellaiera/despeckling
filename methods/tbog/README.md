```bash
$ cd despeckling

$ mamba env create -f methods/tbog/environment.yml

$ conda activate despeckling_tbog

$ pip install -e .

$ tbog \
    --input_path ../dataset/data.npz \
```
