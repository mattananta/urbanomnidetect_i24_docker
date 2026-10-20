FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

ARG DEBIAN_FRONTEND=noninteractive
ARG UOD_REPO=https://github.com/mkturkcan/urbanomnidetect.git
ARG UOD_REF=main

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    YOLO_CONFIG_DIR=/tmp/Ultralytics

RUN apt-get update && apt-get install -y --no-install-recommends \
    git libgl1 libglib2.0-0 libgomp1 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt
RUN git clone --depth 1 --branch "${UOD_REF}" "${UOD_REPO}" /opt/urbanomnidetect

WORKDIR /opt/urbanomnidetect
RUN python -m pip install --upgrade pip && \
    python -m pip install -r requirements.txt && \
    python -m pip install "shapely>=2.0"

COPY evaluate_urbanomnidetect_i24.py /opt/urbanomnidetect/evaluate_urbanomnidetect_i24.py
COPY docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

WORKDIR /opt/urbanomnidetect
ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]
CMD ["--help"]
