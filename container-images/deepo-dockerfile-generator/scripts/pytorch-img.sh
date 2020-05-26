#!/bin/sh

WORKSPACE=/workspace

mkdir -p ${WORKSPACE}/container-images/generated/pytorch-py36-jupyter-cpu

python /deepo/generator/generate.py \
    ${WORKSPACE}/container-images/generated/pytorch-py36-jupyter-cpu/Dockerfile \
    pytorch jupyter python==3.6
