

PHONY: build
build:
	docker build -t pytorchflask .


PHONY: run
run:
	docker run -d -p 127.0.0.1:5000:80 --name pytorchflask pytorchflask


PHONY: rm
rm:
	docker rm -f pytorchflask


PHONY: logs
logs:
	docker logs pytorchflask

