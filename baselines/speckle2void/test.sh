##!/bin/bash

METHOD_NAME="speckle2void"
~/miniforge3/envs/despeckling_${METHOD_NAME}/bin/python test.py --input_path "$1"
