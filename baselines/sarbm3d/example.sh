##!/bin/bash

INPUT_PATH=../../../dataset/input_1.mat
MATLAB_SCRIPT_PATH=../../../SARBM3D_v10_linux64/SARBM3D_v10.m
L=1

./run.sh "$INPUT_PATH" "$MATLAB_SCRIPT_PATH" "" "$L"
