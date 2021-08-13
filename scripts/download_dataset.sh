#!/usr/bin/env bash
mkdir data

wget https://vision.in.tum.de/webshare/u/dendorfp/TrajectoryPredictionData/datasets.zip
unzip datasets.zip -d data/datasets
rm datasets.zip
