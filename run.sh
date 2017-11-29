#!/bin/bash

pip install --user virtualenv

python -m virtualenv ./VENV2.7
source ./VENV2.7/bin/activate

pip install scikit-learn
pip install numpy
pip install scipy

cd code
python main.py

cd ..

rm -fr VENV2.7