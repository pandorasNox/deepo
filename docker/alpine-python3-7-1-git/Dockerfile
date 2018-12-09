FROM  python:3.7.1-alpine3.8

# install git
RUN apk --update add git openssh && \
    rm -rf /var/lib/apt/lists/* && \
    rm /var/cache/apk/*

COPY entrypoint.sh .

ENTRYPOINT [ "/entrypoint.sh" ]
