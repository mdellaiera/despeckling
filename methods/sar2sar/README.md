```bash
$ cd despeckling

$ mamba env create -f methods/sar2sar/environment.yml

$ conda activate despeckling_sar2sar

$ pip install -e .

$ sar2sar \
    --input_path ../dataset/data.npz \
    --project_path ../deepdespeckling/deepdespeckling/sar2sar
```
