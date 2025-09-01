
FROM tensorflow/tensorflow:latest-gpu
# ^or just latest-gpu if you don't need Jupyter


ENV  POETRY_VIRTUALENVS_IN_PROJECT=true \
  POETRY_HOME="/home/poetry" \
  VENV_PATH="/home/ml_app .venv" \
  USER="ml_app"

ENV  PATH="$POETRY_HOME/bin:$VENV_PATH/bin:$PATH"

RUN useradd $USER

RUN apt-get update \
  && DEBIAN_FRONTEND=noninteractive apt-get install -y python3-pip python3-dev \
  && cd /usr/local/bin \
  && ln -s /usr/bin/python3 python \
  && pip3 install --upgrade pip \
  && pip3 install poetry==1.2.0a2

WORKDIR /home/ml_app

COPY pyproject.toml poetry.lock ./
RUN poetry install --without dev

COPY ml_app/app app
COPY ml_app/ml_app.py ./
COPY ml_app/config.py ./
COPY ml_app/entrypoint.sh ./
COPY .env ./
RUN chmod +x entrypoint.sh

ENV FLASK_APP ml_app.py

RUN chown -R ml_app:ml_app /home/ml_app
USER $USER

ENTRYPOINT ["sh", "./entrypoint.sh"]
