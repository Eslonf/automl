FROM ghcr.io/mlflow/mlflow:v3.8.1

ARG MLFLOW_DATA_PATH="/mlflow"

ENV MLFLOW_PORT=5000
ENV MLFLOW_BACKEND_STORE_URI="sqlite://${MLFLOW_DATA_PATH}/mlflow.db"
ENV MLFLOW_ARTIFACT_ROOT="${MLFLOW_DATA_PATH}/artifacts"

RUN mkdir -p ${MLFLOW_ARTIFACT_ROOT} && \
    chmod -R 777 ${MLFLOW_DATA_PATH}

EXPOSE ${MLFLOW_PORT}

CMD sh -c "mlflow server \
     --host 0.0.0.0 \
     --port ${MLFLOW_PORT} \
     --backend-store-uri ${MLFLOW_BACKEND_STORE_URI} \
     --default-artifact-root ${MLFLOW_ARTIFACT_ROOT} \
     --allowed-hosts '*'"