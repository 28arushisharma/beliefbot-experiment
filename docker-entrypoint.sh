#!/bin/sh
# Builds DATABASE_URL from the DB_* variables injected by ECS (password comes
# from the RDS secret and may contain characters that must be URL-encoded).
# An explicit DATABASE_URL, if set, wins.
set -e

if [ -z "$DATABASE_URL" ] && [ -n "$DB_HOST" ]; then
  DATABASE_URL="$(python - <<'EOF'
import os
from urllib.parse import quote

e = os.environ
url = "postgresql://{user}:{pw}@{host}:{port}/{name}".format(
    user=quote(e["DB_USER"], safe=""),
    pw=quote(e["DB_PASSWORD"], safe=""),
    host=e["DB_HOST"],
    port=e.get("DB_PORT", "5432"),
    name=e.get("DB_NAME", "otree"),
)
if e.get("DB_SSLMODE"):
    url += "?sslmode=" + e["DB_SSLMODE"]
print(url)
EOF
)"
  export DATABASE_URL
fi

exec "$@"
