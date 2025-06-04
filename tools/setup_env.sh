#!/bin/bash

# For training
cd ../ms-swift
pip install -e .

cd requirements/
bash install_all.sh

# For evaluation
pip install vllm python_dateutil flash_attn latex2sympy2 word2number Pebble timeout-decorator

