# deepo
custom builds for the deepo docker env's

python:3.7.1-alpine3.8

`docker run -it --rm python:3.7.1-alpine3.8 sh`

`docker build -t python:3.7.1-alpine3.8-git ./docker/alpine-python3-7-1-git/`

`docker run -it --rm python:3.7.1-alpine3.8-git sh`

`docker run -it --rm -v $(pwd):/workspace python:3.7.1-alpine3.8-git sh`

`docker run -it --rm -v $(pwd):/workspace python:3.7.1-alpine3.8-git`

## setup
`docker build -t python:3.7.1-alpine3.8-git ./docker/alpine-python3-7-1-git/`
`docker run -it --rm -v $(pwd):/workspace python:3.7.1-alpine3.8-git`

## example run it
`docker run -it --rm -p 8023:8888 -v $(pwd):/root/hostbooks pandorasnox/deepo:pytorch-py36-jupyter-cpu jupyter notebook --no-browser --ip=0.0.0.0 --allow-root --NotebookApp.token= --notebook-dir='/root'`
