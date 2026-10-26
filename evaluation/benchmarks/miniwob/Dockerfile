FROM python:3.12-bookworm
# Pick up distro ca-certificates updates (e.g. the eMudhra intermediate CA distrust) before base images refresh.
RUN apt-get update && apt-get install -y --no-install-recommends --only-upgrade ca-certificates && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y python3 python3-pip git

RUN git clone https://github.com/Farama-Foundation/miniwob-plusplus.git /miniwob-plusplus && \
    git -C "/miniwob-plusplus" reset --hard 7fd85d71a4b60325c6585396ec4f48377d049838

ENV MINIWOB_URL="file:///miniwob-plusplus/miniwob/html/miniwob/"

# docker build -t xingyaoww/od-eval-miniwob .
