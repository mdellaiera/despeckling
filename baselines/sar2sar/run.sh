##!/bin/bash

METHOD_NAME="sar2sar"

~/miniforge3/envs/despeckling_${METHOD_NAME}/bin/python run.py \
     --input_path "$1" \
     --project_path "$2" \
     ${3:+--output_path "$3"}
