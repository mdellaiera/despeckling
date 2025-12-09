```bash
$ cd despeckling

$ mamba env create -f methods/merlin/environment.yml

$ conda activate despeckling_merlin

$ pip install -e .

$ merlin \
    --input_path ../dataset/data.npz \
    --project_path ../deepdespeckling/deepdespeckling/merlin
```
