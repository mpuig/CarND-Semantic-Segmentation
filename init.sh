#!/bin/bash
git clone https://github.com/mpuig/CarND-Semantic-Segmentation.git
cd CarND-Semantic-Segmentation/data
wget http://kitti.is.tue.mpg.de/kitti/data_road.zip
unzip data_road.zip
cd ..

pip install tqdm
python main.py