```bash
$ cd despeckling

$ mamba env create -f methods/merlin/environment.yml

$ conda activate despeckling_merlin

$ pip install -e .

$ merlin \
    --input_path ../dataset/data_village.npz \
    --output_path methods/merlin/results/output_village_merlin.mat \
    --project_path ../deepdespeckling/deepdespeckling/merlin
```
