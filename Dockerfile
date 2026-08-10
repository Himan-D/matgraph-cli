FROM python:3.11-slim
WORKDIR /app
COPY pyproject.toml README.md ./
COPY matgraph ./matgraph
RUN pip install --no-cache-dir -e .
EXPOSE 8000
ENV MATGRAPH_CACHE_DIR=/data/cache
CMD ["uvicorn","matgraph.graphql_app:app","--host","0.0.0.0","--port","8000"]
