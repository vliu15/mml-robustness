#!/bin/bash

mkdir -p data/celeba

# Download the necessary files
wget https://s3-us-west-1.amazonaws.com/udacity-dlnfd/datasets/celeba.zip -P data/celeba
wget https://raw.githubusercontent.com/vliu15/celeba_files/main/list_attr_celeba.txt -P data/celeba
wget https://raw.githubusercontent.com/vliu15/celeba_files/main/list_eval_partition.txt -P data/celeba

# Unzip images
unzip data/celeba/celeba.zip -d data/celeba
