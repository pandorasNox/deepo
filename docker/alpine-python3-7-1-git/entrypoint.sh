#!/bin/sh

git clone https://github.com/ufoym/deepo.git

cd deepo

git checkout f047376c0893833b2e92db266ebc492192616e17

/workspace/scripts/custom-gen-docker.sh
