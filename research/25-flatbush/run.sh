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

export SCOUT_DEER_STEP_LIMIT=1000
exec .scenery-harness/deer-venv/bin/python -m scene_scout.scenery_trial \
    deer-visual '25 Flatbush Avenue, Brooklyn, New York' 1990 --out "$out"
