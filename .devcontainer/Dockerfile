ARG VARIANT="3.8.12-bullseye"
FROM mcr.microsoft.com/vscode/devcontainers/python:0-${VARIANT}

# Update pip
RUN pip install --upgrade pip

# Install Poetry
RUN curl -sSL https://install.python-poetry.org | python3 -

## Git setup
RUN git config --global diff.tool vscode && git config --global difftool.vscode.cmd 'code --wait --diff $LOCAL $REMOTE'
