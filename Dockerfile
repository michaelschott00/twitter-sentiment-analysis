FROM python:3.12-slim

ENV HERMES_HOME=/home/agent/.hermes
ENV HOME=/home/agent
# ENV HERMES_TUI=1

WORKDIR /workspace

RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates libgomp1 git && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y --no-install-recommends nodejs npm && rm -rf /var/lib/apt/lists/*
RUN curl -fsSL https://hermes-agent.nousresearch.com/install.sh | bash

RUN curl -fsSL https://opencode.ai/install | bash

COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir -r requirements.txt

RUN apt-get install -y unzip \
    && curl -fsSLO https://releases.hashicorp.com/terraform/1.16.0/terraform_1.16.0_linux_amd64.zip \
    && unzip terraform_1.16.0_linux_amd64.zip \
    && mv terraform /usr/bin

RUN useradd -m -u 1000 agent
RUN mkdir -p $HERMES_HOME && chown agent:agent $HERMES_HOME
USER agent
ENTRYPOINT "/bin/bash"
