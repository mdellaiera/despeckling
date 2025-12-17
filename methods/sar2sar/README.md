```bash
$ cd despeckling

$ mamba env create -f methods/sar2sar/environment.yml

$ conda activate despeckling_sar2sar

$ pip install -e .

$ sar2sar \
    --input_path ../dataset/data_village.npz \
    --output_path methods/sar2sar/results/output_village_sar2sar.mat \
    --project_path ../deepdespeckling/deepdespeckling/sar2sar
```
