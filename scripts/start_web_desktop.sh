#!/usr/bin/env bash
# Start a local virtual X11 desktop exposed through browser-based noVNC.
set -euo pipefail

WEB_DISPLAY=${WEB_DISPLAY:-:99}
WEB_RESOLUTION=${WEB_RESOLUTION:-1920x1080x24}
VNC_PORT=${VNC_PORT:-15900}
NOVNC_PORT=${NOVNC_PORT:-6080}
NOVNC_WEB_ROOT=/usr/share/novnc

for executable in Xvfb openbox x11vnc websockify; do
    if ! command -v "$executable" >/dev/null 2>&1; then
        echo "Missing web visualization dependency: $executable" >&2
        echo "Rebuild the development container from docker/Dockerfile." >&2
        exit 1
    fi
done

display_number=${WEB_DISPLAY#:}
display_socket="/tmp/.X11-unix/X${display_number}"

if ! pgrep -f "^Xvfb ${WEB_DISPLAY}( |$)" >/dev/null; then
    Xvfb "$WEB_DISPLAY" \
        -screen 0 "$WEB_RESOLUTION" \
        -ac +extension GLX +render -noreset \
        > /tmp/handeye_xvfb.log 2>&1 &
fi

for _ in $(seq 1 30); do
    [[ -S "$display_socket" ]] && break
    sleep 0.1
done
if [[ ! -S "$display_socket" ]]; then
    echo "Virtual X11 display failed to start; see /tmp/handeye_xvfb.log." >&2
    exit 1
fi

if ! pgrep -f "^openbox --sm-disable( |$)" >/dev/null; then
    DISPLAY="$WEB_DISPLAY" openbox --sm-disable \
        > /tmp/handeye_openbox.log 2>&1 &
fi

if ! pgrep -f "^x11vnc .* -rfbport ${VNC_PORT}( |$)" >/dev/null; then
    x11vnc \
        -display "$WEB_DISPLAY" \
        -forever -shared -localhost -nopw -quiet \
        -rfbport "$VNC_PORT" \
        > /tmp/handeye_x11vnc.log 2>&1 &
fi

if ! pgrep -f "websockify .*127\\.0\\.0\\.1:${NOVNC_PORT} .*127\\.0\\.0\\.1:${VNC_PORT}" >/dev/null; then
    websockify \
        --web="$NOVNC_WEB_ROOT" \
        "127.0.0.1:${NOVNC_PORT}" \
        "127.0.0.1:${VNC_PORT}" \
        > /tmp/handeye_websockify.log 2>&1 &
fi

sleep 1
if ! pgrep -f "^x11vnc .* -rfbport ${VNC_PORT}( |$)" >/dev/null; then
    echo "VNC server failed to start; see /tmp/handeye_x11vnc.log." >&2
    exit 1
fi
if ! pgrep -f "websockify .*127\\.0\\.0\\.1:${NOVNC_PORT} .*127\\.0\\.0\\.1:${VNC_PORT}" >/dev/null; then
    echo "noVNC bridge failed to start; see /tmp/handeye_websockify.log." >&2
    exit 1
fi

echo "Web visualization: http://localhost:${NOVNC_PORT}/vnc.html?autoconnect=1&resize=scale"
