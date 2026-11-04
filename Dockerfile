# Place in the root of your oTree project (next to settings.py), along with
# docker-entrypoint.sh. requirements.txt must include `otree` and
# `psycopg2-binary`.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY --chmod=755 docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
COPY . .

RUN useradd --create-home otree && chown -R otree /app
USER otree

EXPOSE 8000
ENTRYPOINT ["docker-entrypoint.sh"]
CMD ["otree", "prodserver", "8000"]
