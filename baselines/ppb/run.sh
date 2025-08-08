##!/bin/bash

METHOD_NAME="ppb"

~/miniforge3/envs/despeckling_${METHOD_NAME}/bin/python run.py \
    --input_path "$1" \
    --matlab_script_path "$2" \
    ${3:+--output_path "$3"} \
    ${4:+--L "$4"} \
    ${5:+--hw "$5"} \
    ${6:+--hd "$6"} \
    ${7:+--alpha "$7"} \
    ${8:+--T "$8"} \
    ${9:+--nbits "$9"} \
    ${10:+--estimate_path "${10}"}
