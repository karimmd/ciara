FROM python:3.12-slim
RUN apt-get update -qq \
 && apt-get install -y -qq --no-install-recommends iproute2 iputils-ping procps curl \
 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
