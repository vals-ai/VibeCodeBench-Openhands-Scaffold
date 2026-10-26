FROM python:3.12-bookworm
# Pick up distro ca-certificates updates (e.g. the eMudhra intermediate CA distrust) before base images refresh.
RUN apt-get update && apt-get install -y --no-install-recommends --only-upgrade ca-certificates && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y python3 python3-pip

RUN mkdir /workspace
WORKDIR /workspace


COPY data/ /workspace/data/
COPY tools/ /workspace/tools/

# TODO: NEED TO FIGURE DEPENDECIES FOR THESE TOOLS

# pushd evaluation/toolqa
# mkdir data
# python3 -c "from utils import download_data, download_tools; download_data('/workspace'); download_tools('/workspace')"
# docker build --network host -t xingyaoww/od-eval-toolqa .
