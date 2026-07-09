# python:3.11-slim, not -alpine: psycopg2-binary (the `postgres` extra) ships
# a manylinux wheel for glibc, so slim needs no libpq-dev/gcc build stage —
# alpine's musl libc would force a compile-from-source path instead.
FROM python:3.11-slim

WORKDIR /app

COPY . /app

# Editable install is deliberate, not a shortcut. `-e .` makes pip write a
# .pth file pointing at /app (the project root) instead of copying form4lab/
# into site-packages, so:
#   1. __file__ inside form4lab/cli.py still resolves to /app/form4lab/cli.py,
#      which is what init-db's alembic.ini lookup
#      (Path(__file__).resolve().parent.parent / "alembic.ini") depends on to
#      find /app/alembic.ini at runtime;
#   2. /app itself lands on sys.path, so docker-compose.yml's
#      ./strategies:/app/strategies bind mount is importable as a top-level
#      `strategies` namespace package (no PYTHONPATH wrangling needed) for
#      STRATEGY_PATH=strategies.my_module:MyClass.
# This sidesteps the wheel-packaging gap a non-editable install would hit
# (a built wheel only contains the `form4lab` package unless
# pyproject.toml's [tool.hatch.build.targets.wheel.force-include] also ships
# alembic.ini/alembic/ — included for non-editable (wheel) installs, not exercised by
# this editable Docker install).
RUN pip install --no-cache-dir -e ".[trade,postgres]"

RUN chmod +x docker/entrypoint.sh

ENTRYPOINT ["/app/docker/entrypoint.sh"]
CMD ["web"]
