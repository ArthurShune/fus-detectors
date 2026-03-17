FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl ca-certificates bzip2 && \
    rm -rf /var/lib/apt/lists/*

ENV MAMBA_ROOT_PREFIX=/opt/micromamba
ENV MAMBA_NO_BANNER=1
RUN mkdir -p ${MAMBA_ROOT_PREFIX} && \
    curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | \
    tar -xvj -C /usr/local/bin/ --strip-components=1 bin/micromamba
WORKDIR /app
COPY environment.yml /app/environment.yml
RUN micromamba create -y -n fus-detectors -f environment.yml && micromamba clean -a -y
COPY . /app
ENV PYTHONPATH=/app
SHELL ["/bin/bash", "-lc"]
CMD ["micromamba", "run", "-n", "fus-detectors", "bash"]
