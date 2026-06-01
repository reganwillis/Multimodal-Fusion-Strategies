#!/bin/bash

# install dependencies
# TODO: only create new venv if this dir does not exist
#python3 -m venv venv
source venv/bin/activate
#pip install -r requirements.txt

# run training scripts
#    parser.add_argument('--arch', type=str, required=True)
#    parser.add_argument('--path-mobilenet', type=str, required=False)
#    parser.add_argument('--path-bert', type=str, required=False)
#    parser.add_argument('--debug', type=bool, default=False, required=False)

python3 train.py --arch 'MobileNetV2' --debug True
exit
python3 train/late-fusion.py
python3 train/mid-fusion.py
python3 train/early-fusion.py
python3 train/mid-fusion-vit.py
python3 train/early-fusion-vit.py
