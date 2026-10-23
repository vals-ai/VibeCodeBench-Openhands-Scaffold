FROM python:3.12-bookworm
# Pick up distro ca-certificates updates (e.g. the eMudhra intermediate CA distrust) before base images refresh.
RUN apt-get update && apt-get install -y --no-install-recommends --only-upgrade ca-certificates && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y python3 python3-pip git gcc

WORKDIR /root

COPY requirements.txt .
RUN pip install -r requirements.txt

# docker build -t xingyaoww/od-eval-mint:v1.0 .
