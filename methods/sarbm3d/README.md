```bash
$ cd despeckling

$ mamba env create -f methods/sarbm3d/environment.yml

$ conda activate despeckling_sarbm3d

$ pip install -e .

$ sarbm3d \
    --input_path ../dataset/data_village.npz \
    --output_path methods/sarbm3d/results/output_village_sarbm3d.mat \
    --matlab_script_path ../SARBM3D_v10_linux64/SARBM3D_v10.m \
    --L 1
```
