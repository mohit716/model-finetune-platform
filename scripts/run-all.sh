#!/usr/bin/env bash
# Start Redis (if needed), Celery worker, then API. Stop with Ctrl+C stops both.
# Run from repo root: ./scripts/run-all.sh
set -e
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

export PATH="$REPO_ROOT/.venv/bin:$PATH"

# 1. Redis
REDIS_CLI="$REPO_ROOT/redis-build/redis-stable/src/redis-cli"
if [[ -x "$REDIS_CLI" ]]; then
  if ! "$REDIS_CLI" ping 2>/dev/null; then
    echo "Starting Redis..."
    "$REPO_ROOT/scripts/start-redis.sh" || true
  fi
fi

# 2. Celery worker in background
echo "Starting Celery worker..."
celery -A app.workers.celery_app worker --loglevel=info --concurrency=1 &
CELERY_PID=$!

cleanup() {
  echo "Stopping Celery worker (PID $CELERY_PID)..."
  kill $CELERY_PID 2>/dev/null || true
  wait $CELERY_PID 2>/dev/null || true
  exit 0
}
trap cleanup SIGINT SIGTERM

# 3. API in foreground (Cursor "Open in browser" works here)
echo "Starting API on http://0.0.0.0:8000 ..."
exec uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
