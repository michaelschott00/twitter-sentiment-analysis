FROM python:3.12-slim

ENV HERMES_HOME=/hermes
# ENV HERMES_TUI=1

RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y --no-install-recommends nodejs npm && rm -rf /var/lib/apt/lists/*
RUN curl -fsSL https://hermes-agent.nousresearch.com/install.sh | bash

RUN curl -fsSL https://opencode.ai/install | bash

WORKDIR /workspace

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN useradd -m -u 1000 agent
RUN mkdir -p /hermes && chown agent:agent /hermes
USER agent
ENTRYPOINT "/bin/bash"
