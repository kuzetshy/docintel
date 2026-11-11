-- Runs once, on the first start of an empty Postgres volume.
-- The app database (POSTGRES_DB) is created by the image itself;
-- Langfuse gets its own database on the same server.
CREATE DATABASE langfuse;
