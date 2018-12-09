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
