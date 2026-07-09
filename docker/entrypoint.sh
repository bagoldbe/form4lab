#!/usr/bin/env bash
set -euo pipefail

# Migration ownership: the `web` service is the ONLY service that runs
# `form4lab init-db` (an idempotent `alembic upgrade head`) on startup. The
# `scheduler` service never runs migrations itself — docker-compose.yml
# declares it `depends_on: web: condition: service_healthy`, and web's
# healthcheck is an HTTP request that requires a working DB query, so the
# scheduler container only starts once the schema is provably at head. This
# keeps "who owns the schema" unambiguous instead of racing two `alembic
# upgrade` calls against the same fresh database.
#
# Known limitation (not handled here): a database created by the OLD
# create_all-based init-db (pre-Alembic)
# has no alembic_version table, so a later `alembic upgrade head` against it
# would try to re-create every already-existing table and fail. Recovering
# that database would need a one-time manual `alembic stamp head` to mark
# the existing schema as current. Not handled by this script because no such
# database exists for a fresh install — `docker compose up` always starts
# `db` from an empty `pgdata` volume.

cmd="${1:-}"

if [ "$cmd" = "web" ]; then
  form4lab init-db
  exec uvicorn form4lab.main:app --host 0.0.0.0 --port 8000
elif [ "$cmd" = "scheduler" ]; then
  exec form4lab scheduler
else
  exec form4lab "$@"
fi
