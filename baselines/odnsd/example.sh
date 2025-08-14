##!/bin/bash

INPUT_PATH_SAR=../../../dataset/input_1.mat
INPUT_PATH_OPT=../../../dataset/input_1.mat
MATLAB_SCRIPT_PATH=../../../SARBM3D_v10_linux64/SARBM3D_v10.m

./run.sh "$INPUT_PATH_SAR" "$INPUT_PATH_OPT" "$MATLAB_SCRIPT_PATH"
