#!/usr/bin/env bash
# Start local Redis (no sudo). Run from repo root: ./scripts/start-redis.sh
set -e
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REDIS_SERVER="$REPO_ROOT/redis-build/redis-stable/src/redis-server"
REDIS_DIR="$REPO_ROOT/data/redis"

if [[ ! -x "$REDIS_SERVER" ]]; then
  echo "Redis not built. Build with: cd redis-build && curl -sL https://download.redis.io/redis-stable.tar.gz -o redis-stable.tar.gz && tar xzf redis-stable.tar.gz && cd redis-stable && make"
  exit 1
fi

mkdir -p "$REDIS_DIR"
"$REDIS_SERVER" --daemonize yes --dir "$REDIS_DIR" --port 6379
echo "Redis started on port 6379."
