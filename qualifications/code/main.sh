#!/bin/sh
echo "generate process data...."
python3 ../feature/generate.py
echo "train model ...."
python3 ../model/basic_model.py
echo "inference ...."
python3 ./main.py
