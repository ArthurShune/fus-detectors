FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    git wget curl ca-certificates build-essential python3 python3-venv python3-pip && \
    rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY . /app
CMD ["/bin/bash"]
