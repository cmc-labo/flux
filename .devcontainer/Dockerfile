FROM mcr.microsoft.com/devcontainers/rust:1

# Install additional packages if needed
RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
    && apt-get -y install --no-install-recommends build-essential libssl-dev pkg-config

# Set the default shell to bash
ENV SHELL=/bin/bash
