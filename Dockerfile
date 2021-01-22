# For more information, please refer to https://aka.ms/vscode-docker-python
FROM boa50/python-env

WORKDIR /app
ADD . /app

# Switching to a non-root user, please refer to https://aka.ms/vscode-docker-python-user-rights
RUN chown -R boa50 /app