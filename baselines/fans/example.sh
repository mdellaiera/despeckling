##!/bin/bash

INPUT_PATH=../../../dataset/input_1.mat
MATLAB_SCRIPT_PATH=../../../FANS_v10_linux64/FANS.m
L=1

./run.sh "$INPUT_PATH" "$MATLAB_SCRIPT_PATH" "" "$L"
