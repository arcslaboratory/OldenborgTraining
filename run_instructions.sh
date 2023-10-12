#!/usr/bin/env bas
python training.py perfect-static resnet18 --pretrained --gpu 2 --local_data data/PerfectStaticTextures/
python training.py perfect-random resnet18 --pretrained --gpu 2 --local_data data/PerfectRandomTextures/
python training.py wandering-static resnet18 --pretrained --gpu 2 --local_data data/WanderingStaticTextures/
python training.py wandering-random resnet18 --pretrained --gpu 2 --local_data data/WanderingRandomTextures/