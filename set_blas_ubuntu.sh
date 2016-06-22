#!/usr/bin/env bash
echo -e "\n[global]\nfloatX=float32\n" >> ~/.theanorc
sudo update-alternatives --config libblas.so.3
python mnist_cnn.py

