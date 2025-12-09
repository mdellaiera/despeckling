```bash
$ cd despeckling

$ mamba env create -f methods/ppb/environment.yml

$ conda activate despeckling_ppb

$ pip install -e .

$ ppb \
    --input_path ../dataset/data.npz \
    --matlab_script_path ../ppb/ppbNakagami/ppb_nakagami.m \
    --L 1 \
    --hw 10 \
    --hd 3 \
    --alpha 0.92 \
    --T 0.2 \
    --nbits 4
```