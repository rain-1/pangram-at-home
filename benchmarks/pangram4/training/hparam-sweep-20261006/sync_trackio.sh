#!/bin/bash
# Off-host Trackio sync: the training Space has no Hub token, so runs log to a local SQLite database there.
# This loop (run on a machine logged in to the Hub) snapshots that database over `hf spaces ssh` and syncs it
# to a private Trackio Space (created once with `trackio sync --private`). Usage: sync_trackio.sh [space_id] [interval_seconds, 0 = once]
set -u
PROJECT=${PROJECT:-pangram-hparam-sweep-20261006}
SPACE_ID=${1:-eac123/pangram-hparam-sweep-trackio}
EVERY=${2:-600}
REMOTE=${REMOTE:-/tmp/pangram-hparam-sweep-20261006/trackio}
export TRACKIO_DIR=${TRACKIO_DIR:-$HOME/.cache/pangram-trackio}
mkdir -p "$TRACKIO_DIR"
SSH=$(hf spaces ssh --dry-run open-text-detector/training 2>/dev/null | tail -1)
while true; do
  # sqlite's online backup gives a consistent copy while trainers are writing.
  if $SSH -o BatchMode=yes "cd $REMOTE && python -c \"import sqlite3;s=sqlite3.connect('$PROJECT.db');d=sqlite3.connect('/tmp/trackio-snapshot.db');s.backup(d);d.close()\" && cat /tmp/trackio-snapshot.db" \
      > "$TRACKIO_DIR/$PROJECT.db.part" && [ -s "$TRACKIO_DIR/$PROJECT.db.part" ]; then
    mv "$TRACKIO_DIR/$PROJECT.db.part" "$TRACKIO_DIR/$PROJECT.db"
    # The Gradio dashboard reads the bucket copy only at startup, and `trackio sync` then waits on it and times out.
    # So upload the database to the Space's bucket directly and restart the dashboard to reload it.
    uv run -q --with trackio==0.40.0 python -c "from trackio.deploy import upload_project_to_bucket as u; u('$PROJECT', '$SPACE_ID-bucket')" \
      && hf spaces restart "$SPACE_ID" > /dev/null \
      && echo "$(date -u +%H:%M:%S) synced" || echo "$(date -u +%H:%M:%S) sync failed"
  else
    echo "$(date -u +%H:%M:%S) snapshot failed"
  fi
  [ "$EVERY" = 0 ] && break  # interval 0: sync once
  sleep "$EVERY"
done
