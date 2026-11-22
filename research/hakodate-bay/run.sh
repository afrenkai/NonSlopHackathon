#!/bin/sh
set -eu

usage() {
    echo "Usage: $0 NEW_OUTPUT_DIRECTORY"
}

if [ "${1:-}" = '--help' ]; then
    usage
    exit 0
fi
if [ "$#" -ne 1 ] || [ -z "$1" ]; then
    usage >&2
    exit 2
fi

case "$1" in
    /*) out="$1" ;;
    *) out="$PWD/$1" ;;
esac
here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
scout_root=${SCOUT_ROOT:-"$here/../../../nonslophackathon"}
cd "$scout_root" || exit 1
if [ ! -x .scenery-harness/deer-venv/bin/python ]; then
    echo 'Missing the DeerFlow virtual environment in SCOUT_ROOT. See README.md.' >&2
    exit 1
fi

# The protocol only takes (location, year); "December" rides inside the
# location string so the task reads "...of <location> in 2023".
# RUN_SECONDS is pinned at 900 upstream; patch it to 30 min here instead of
# editing the pinned protocol module.
export SCOUT_DEER_STEP_LIMIT=1000
exec .scenery-harness/deer-venv/bin/python -c '
import sys
from scene_scout import scenery_protocol
scenery_protocol.RUN_SECONDS = 1800
sys.argv = ["scene_scout.scenery_trial", "deer-visual",
            "41.767, 140.717 Hakodate Bay waterfront (Kanemori Red Brick Warehouse area), Hakodate, Hokkaido, Japan, during December",
            "2023", "--out", sys.argv[1]]
from scene_scout.scenery_trial import main
raise SystemExit(main())
' "$out"
