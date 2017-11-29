#!/bin/bash

virtualenv ./VENV2.7
source ./VENV2.7/bin/activate

pip install scikit-learn
pip install numpy

cd code
python main.py

cd ..

rm -fr VENV2.7