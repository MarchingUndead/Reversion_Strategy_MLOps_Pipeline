# Unified runtime image for the reversion pipeline. One image powers
# Streamlit and every Airflow service (init, webserver, scheduler) — the base
# is `apache/airflow:2.10.3` so airflow CLI and its entrypoint script are
# available without a second image. Pipeline runtime deps are pulled from the
# SoT `requirements.txt`. The streamlit/fastapi/uvicorn/prometheus-client
# packages travel with the image so the same artefact can serve the model and
# render the UI; airflow services just override `command:` in compose.

FROM apache/airflow:2.10.3

USER airflow
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Bake the project source into /app so the streamlit container needs no host
# mount. Airflow services mount src/ / scripts/ / config.yaml at
# /opt/airflow/project/ instead (see docker-compose.yml volumes).
USER root
RUN mkdir -p /app && chown airflow:0 /app
COPY --chown=airflow:0 src /app/src
COPY --chown=airflow:0 config.yaml /app/config.yaml

USER airflow

# Don't override the base image's WORKDIR (/opt/airflow) — airflow expects it.
# Streamlit's CMD uses an absolute path so it doesn't depend on cwd.
EXPOSE 8501
CMD ["streamlit", "run", "/app/src/streamlit_app.py", \
     "--server.address=0.0.0.0", "--server.port=8501", \
     "--server.headless=true"]
