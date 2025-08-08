##!/bin/bash

INPUT_PATH=../../../dataset/input_1.mat
CHECKPOINT_PATH=../../../speckle2void/s2v_checkpoint/model.ckpt-299999
LIBRARIES_PATH=../../../speckle2void/libraries
NORM=100

./run.sh "$INPUT_PATH" "$CHECKPOINT_PATH" "$LIBRARIES_PATH" "$NORM"
