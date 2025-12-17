```bash
$ cd despeckling

$ mamba env create -f methods/speckle2void/environment.yml

$ conda activate despeckling_speckle2void

$ pip install -e .

$ speckle2void \
    --input_path ../dataset/data_village.npz \
    --output_path methods/speckle2void/results/output_village_speckle2void.mat \
    --checkpoint_path ../speckle2void/s2v_checkpoint/model.ckpt-299999 \
    --libraries_path ../speckle2void/libraries \
    --norm 100
```
