#!/usr/bin/env bash
# Run a command; if it fails, surface the last 60 lines of output as a GitHub Actions error annotation.
set -o pipefail
LOG=$(mktemp)
"$@" 2>&1 | tee "$LOG"
rc=${PIPESTATUS[0]}
if [ "$rc" -ne 0 ]; then
  python3 - "$LOG" "$*" <<'PY'
import sys
tail = open(sys.argv[1], errors="replace").read().splitlines()[-60:]
msg = "\n".join(tail).replace("%", "%25").replace("\r", "").replace("\n", "%0A")
print(f"::error title=Failed: {sys.argv[2][:80]}::{msg}")
PY
fi
exit "$rc"
