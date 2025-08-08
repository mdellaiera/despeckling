##!/bin/bash

METHOD_NAME="bm3d"

~/miniforge3/envs/despeckling_${METHOD_NAME}/bin/python run.py \
    --input_path "$1" \
    --sigma_psd "$2" \
    ${3:+--output_path "$3"}
