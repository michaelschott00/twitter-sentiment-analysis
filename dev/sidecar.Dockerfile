FROM python:3.12-slim

WORKDIR /workspace

RUN apt-get update && apt-get install -y gh ssh

# Install basic utils
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl ca-certificates git unzip \
    && rm -rf /var/lib/apt/lists/*

# Install terraform
RUN curl -sSLO https://releases.hashicorp.com/terraform/1.16.1/terraform_1.16.1_linux_amd64.zip \
    && unzip terraform_1.16.1_linux_amd64.zip \
    && mv terraform /usr/local/bin

# Install azure-cli
RUN pip install --no-cache-dir azure-cli "mcp[cli]"

# Install azcopy
RUN curl -sSLO https://aka.ms/downloadazcopy-v10-linux \
    && tar -xvf downloadazcopy-v10-linux \
    && mv azcopy_linux_amd64_*/azcopy /usr/local/bin

# Add non-privileged user
RUN useradd -m -u 1000 agent
USER agent

ENTRYPOINT ["python", "dev/mcp_server.py"]
