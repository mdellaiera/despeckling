```bash
$ cd despeckling

$ mamba env create -f methods/fans/environment.yml

$ conda activate despeckling_fans

$ pip install -e .

$ fans \
    --input_path ../dataset/data.npz \
    --matlab_script_path ../FANS_v10_linux64/FANS.m \
    --L 1
```
