##!/bin/bash

METHOD_NAME="bm3d"
~/miniforge3/envs/despeckling_${METHOD_NAME}/bin/python test.py --input_path "$1"
