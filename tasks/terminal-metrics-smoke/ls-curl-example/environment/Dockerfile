FROM ubuntu:24.04

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN printf 'alpha smoke sentinel\n' > alpha-sentinel.txt \
    && printf 'beta smoke sentinel\n' > beta-sentinel.txt
