# syntax=docker/dockerfile:1.4
FROM --platform=$BUILDPLATFORM python:3.10-buster AS builder

WORKDIR /app

# Copy the requirements.txt file to the container
COPY requirements.txt /app

# Install Python dependencies
RUN --mount=type=cache,target=/root/.cache/pip \
    pip3 install -r requirements.txt

# Update APT and install necessary system dependencies
RUN apt-get update && \
    apt-get install -y \
        tdsodbc \
        unixodbc-dev \
        unixodbc-bin \
        openssl \
        apt-transport-https \
        curl \
        bash \
        build-essential \
        gnupg && \
    curl https://packages.microsoft.com/keys/microsoft.asc | apt-key add - && \
    curl https://packages.microsoft.com/config/debian/10/prod.list > /etc/apt/sources.list.d/mssql-release.list && \
    apt-get update && \
    ACCEPT_EULA=Y apt-get install -y msodbcsql17 msodbcsql18 && \
    apt-get clean -y

# Upgrade pip3
RUN pip3 install --upgrade pip

# Configure environment for /bin/bash to use msodbcsql17
RUN echo 'export PATH="$PATH:/opt/mssql-tools/bin"' >> ~/.bash_profile && \
    echo 'export PATH="$PATH:/opt/mssql-tools/bin"' >> ~/.bashrc

# Copy the rest of the application code to the container
COPY . /app

# Download NLTK Data
RUN python3 -m nltk.downloader punkt

ENTRYPOINT ["python3"]
CMD ["app.py"]

FROM builder as dev-envs

RUN apt-get update && \
    apt-get install -y git

RUN groupadd -r docker && \
    useradd -r -s /bin/bash -g docker vscode

# Install Docker tools (cli, buildx, compose)
COPY --from=gloursdocker/docker / /