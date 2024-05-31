#! /bin/bash

pip install pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.0"
mim install mmdet

# mim download mmdet --config rtmdet_tiny_8xb32-300e_coco --dest # for test.