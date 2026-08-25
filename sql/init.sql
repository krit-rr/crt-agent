-- Runs once on first `docker compose up`.
-- LangFuse owns the `langfuse` database; benchmark results go somewhere it can't
-- clobber, so `docker compose down -v` on the tracing stack never eats your data.
CREATE DATABASE crt_db;
GRANT ALL PRIVILEGES ON DATABASE crt_db TO crt;
