```bash
$ cd despeckling

$ mamba env create -f methods/fans/environment.yml

$ conda activate despeckling_fans

$ pip install -e .

$ fans \
    --input_path ../dataset/data_village.npz \
    --output_path methods/fans/results/output_village_fans.mat \
    --matlab_script_path ../FANS_v10_linux64/FANS.m \
    --L 1
```
