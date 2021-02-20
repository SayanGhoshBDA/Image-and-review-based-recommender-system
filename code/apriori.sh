#!/bin/bash

# downloading data
gdown --id 1UwRWoCKUvhwo8N-6MKa6vJNri6QN2nM-
gdown --id 1BM_CPc1XEj1KxJ5Dz5Y2LHi2tkw-ogfp
gdown --id 1X0Iwb5bT-6g0-j3WW0sY55d0qbq6sfw4
gdown --id 1icOcW7sLQGkU0Y-gJdJNya9W0Al3UCX8
unzip -qq product_images.zip

# cloning pytorch-ssim
git clone https://github.com/Po-Hsun-Su/pytorch-ssim.git
cp -r pytorch-ssim/pytorch_ssim/ ./

# installing few modules
pip install importlib-metadata
pip install transformers
pip install pytorch_pretrained_bert
pip install sentencepiece
