#!/usr/bin/env bash
set -eo pipefail

backend="${1:-sim}"
if [[ $# -gt 0 ]]; then
    shift
fi

case "$backend" in
    sim|simulation)
        exec /workspace/scripts/start_simulation.sh "$@"
        ;;
    real|hardware)
        exec /workspace/scripts/start_real_environment.sh "$@"
        ;;
    *)
        echo "用法: ./scripts/start_environment.sh {sim|real} [后端参数...]" >&2
        exit 2
        ;;
esac
