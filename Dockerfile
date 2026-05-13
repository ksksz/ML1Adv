FROM python:3.11-slim

ENV POETRY_VERSION=2.3.2 \
    POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_CREATE=false

WORKDIR /app

RUN pip install "poetry==${POETRY_VERSION}"

COPY pyproject.toml poetry.lock README.md ./
COPY ml1adv_cirrhosis ./ml1adv_cirrhosis
COPY model.py ./

RUN poetry install --only main

COPY notebooks ./notebooks

RUN mkdir -p /app/data /app/model

ENTRYPOINT ["python", "model.py"]
