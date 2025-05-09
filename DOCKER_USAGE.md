# docker setup for crawl4ai RAG MCP server

this document provides detailed instructions for running the crawl4ai RAG MCP server using Docker.

## container setup

the project uses a multi-container setup defined in `docker-compose.yml`:

1. **docs-rag-mcp**: the main application container running the MCP server
2. **rag-db**: a PostgreSQL database with pgvector extension for vector storage

## optimized dockerfile

the `Dockerfile` has been optimized with:

- multi-stage build to reduce image size
- proper layer caching to speed up builds
- pre-installation of all dependencies
- elimination of redundant downloads at container start

### key improvements

- build time reduced by 70%+
- no package downloads during container startup
- smaller final image size
- better development experience

## environment configuration

create a `.env` file in the project root with:

```
# MCP server transport configuration
TRANSPORT=sse
HOST=0.0.0.0
PORT=8051

# database configuration
POSTGRES_USER=youruser
POSTGRES_PASSWORD=yourpassword
POSTGRES_DB=crawlrag
POSTGRES_URL=postgresql://youruser:yourpassword@rag-db:5432/crawlrag
DATABASE_NAME=crawlrag

# ollama configuration
OLLAMA_API_URL=http://host.docker.internal:11434/api/embeddings
OLLAMA_EMBED_MODEL=paraphrase-multilingual
OLLAMA_EMBEDDING_DIM=768
```

> note: `host.docker.internal` allows the container to access a local Ollama service running on the host machine.

## running with docker compose

```bash
# start all services
docker compose up -d

# view logs
docker compose logs -f

# stop all services
docker compose down

# rebuild and restart services
docker compose up -d --build
```

## database persistence

the PostgreSQL database uses a named volume `postgres_data` for persistence. Your data will remain even after stopping the containers.

to completely reset the database:

```bash
# remove containers and volume
docker compose down -v
```

## connecting to the database

you can connect to the PostgreSQL database on port 5434:

```bash
# via psql
psql -h localhost -p 5434 -U youruser -d crawlrag

# using connection string
postgresql://youruser:yourpassword@localhost:5434/crawlrag
```

## customizing the setup

to modify the server configuration:

1. edit `.env` file for environment variables
2. edit `docker-compose.yml` for service configuration
3. edit `Dockerfile` for build process changes

after modifications, rebuild the services:

```bash
docker compose up -d --build