```bash
$ cd despeckling

$ mamba env create -f methods/sarbm3d/environment.yml

$ conda activate despeckling_sarbm3d

$ pip install -e .

$ sarbm3d \
    --input_path ../dataset/data.npz \
    --matlab_script_path ../SARBM3D_v10_linux64/SARBM3D_v10.m \
    --L 1
```
