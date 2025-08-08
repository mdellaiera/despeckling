##!/bin/bash

METHOD_NAME="speckle2void"

~/miniforge3/envs/despeckling_${METHOD_NAME}/bin/python run.py \
    --input_path "$1" \
    --checkpoint_path "$2" \
    --libraries_path "$3" \
    --norm "$4" \
    ${5:+--output_path "$5"}
