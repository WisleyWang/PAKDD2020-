#!/bin/sh
echo "generate process data...."
python3 ../feature/generate.py
echo "train model ...."
python3 ../model/basic_tag_model.py
python3 ../model/lgb_regressionModel.py
echo "inference ...."
python3 ./main.py
