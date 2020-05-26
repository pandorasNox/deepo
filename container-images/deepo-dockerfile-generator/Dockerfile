FROM  python:3.7.7-alpine3.11

# install git
RUN apk --update add git openssh && \
    rm -rf /var/lib/apt/lists/* && \
    rm /var/cache/apk/*

COPY scripts /scripts

ENV DEEPO_GIT_VERSION_HASH 602f238899f6aa8578e62ff2d8f59ec53714ff9a
RUN git clone https://github.com/ufoym/deepo.git && \
    cd deepo && \
    git checkout ${DEEPO_GIT_VERSION_HASH}

ENV WORKDIR workdir

ENTRYPOINT [ "sh" ]
