##!/bin/bash

METHOD_NAME="sarbm3d"

~/miniforge3/envs/despeckling_${METHOD_NAME}/bin/python run.py \
    --input_path "$1" \
    --matlab_script_path "$2" \
    ${3:+--output_path "$3"} \
    ${4:+--L "$4"} 
