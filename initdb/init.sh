#!/usr/bin/env bash
set -euo pipefail

echo "Processing and executing crawled_pages.sql template..."
sed "s/:\(embedding_dim\)/$OLLAMA_EMBEDDING_DIM/g" /docker-entrypoint-initdb.d/crawled_pages.sql.in | psql -v ON_ERROR_STOP=1 -U "$POSTGRES_USER" -d "$POSTGRES_DB"