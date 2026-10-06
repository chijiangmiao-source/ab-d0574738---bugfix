FROM python:3.11-slim

# 纯标准库实现，无第三方依赖，无需联网安装包。
WORKDIR /app

COPY app ./app
COPY scripts ./scripts
COPY tests ./tests

ENV PORT=8080 \
    STATE_FILE=/data/state.json \
    PYTHONUNBUFFERED=1

RUN mkdir -p /data
VOLUME ["/data"]
EXPOSE 8080

# shell form 以便展开 ${PORT}；端口可通过 Compose / 环境变量配置
CMD python3 -m app.server --port ${PORT:-8080} --state ${STATE_FILE:-/data/state.json}
