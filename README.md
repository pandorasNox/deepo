# deepo

custom builds for the deepo docker env's


## workflow

before we can run the build of the deepo custom image we need to generate the dockerfile it.

for that also use docker and a docker generate-deepo-dockerfile image, which we also have to build

### 1. build deepo Dockerfile generator

#### manually
```
# build generate image
docker build -t deepo-dockerfile-generator ./container-images/deepo-dockerfile-generator
```

### 2. generate custom deepo dockerfile

#### manually
```
docker run -it --rm -v $(pwd):/workspace deepo-dockerfile-generator -c "./scripts/pytorch-img.sh"
```

### 3. build custom deepo image (locally)
```
# custom pytorch jupyter image
docker build -t locally/deepo:pytorch-cpu ./container-images/generated/pytorch-py36-jupyter-cpu
```

### 4. run custom deepo image
```
# locally image
docker run -d -t --rm --name custom_deepo_example -p 63824:8888 -v $(PWD):/workspace -w "/workspace" locally/deepo:pytorch-cpu 2> /dev/null || true
```

```
# dockerhub image
docker run -d -t --rm --name custom_deepo_example -p 63824:8888 -v $(PWD):/workspace -w "/workspace" pandorasnox/deepo:pytorch-cpu-0.1.1 2> /dev/null || true
```

```
# run jupyter notebook
docker exec -it custom_deepo_example jupyter notebook --no-browser --ip=0.0.0.0 --allow-root --NotebookApp.token= --notebook-dir='/workspace'
```

### 5. hub.docker build
```
git tag -a "0.1.1" -m ""
git push --tags
```

## todo
- python generator/generate.py ../docker/Dockerfile.tensorflow-py36-cpu tensorflow=1.13.1 python==3.6

- pytorch=1.5.0

- jupyter=5.7.9

- tensorflow=2.1.1


