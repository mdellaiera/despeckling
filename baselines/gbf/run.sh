##!/bin/bash

METHOD_NAME="gbf"

~/miniforge3/envs/despeckling_${METHOD_NAME}/bin/python run.py \
    --input_path_sar "$1" \
    --input_path_opt "$2" \
    --matlab_script_path "$3" \
    ${4:+--output_path "$4"} \
    ${5:+--L "$5"} \
    ${6:+--window_size "$6"} \
    ${7:+--lambda_S "$7"} \
    ${8:+--lambda_RO "$8"} \
    ${9:+--lambda_RS "$9"} \
    ${10:+--N "${10}"} \
    ${11:+--gamma "${11}"} \
    ${12:+--a0 "${12}"}
