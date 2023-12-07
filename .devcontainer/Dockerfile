# For more information, please refer to https://aka.ms/vscode-docker-python
ARG VARIANT="3.8"
FROM mcr.microsoft.com/vscode/devcontainers/python:0-${VARIANT}

# This image includes a non-root user with sudo access. Use 
# the "remoteUser" property in devcontainer.json to use it.
# The user's name is automatically set to vscode.

# [Optional] If your pip requirements rarely change, uncomment this section to add them to the image.
#
# COPY requirements.txt /tmp/pip-tmp/
# RUN pip3 --disable-pip-version-check --no-cache-dir install -r /tmp/pip-tmp/requirements.txt \
#    && rm -rf /tmp/pip-tmp

# [Optional] Uncomment this section to install additional OS packages.
# RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
#    && apt-get -y install --no-install-recommends <your-package-list-here>

# [Optional] Uncomment this line to install global node packages.
# RUN su vscode -c "umask 0002 && . /usr/local/share/nvm/nvm.sh && npm install -g <your-package-list-here>"
