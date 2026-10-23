FROM python:3.12-bookworm
# Pick up distro ca-certificates updates (e.g. the eMudhra intermediate CA distrust) before base images refresh.
RUN apt-get update && apt-get install -y --no-install-recommends --only-upgrade ca-certificates && rm -rf /var/lib/apt/lists/*

RUN pip install scitools-pyke

# docker build -t xingyaoww/od_logic_reasoning .
