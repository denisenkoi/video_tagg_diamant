#!/bin/bash
# Web Server Monitor for Video Tagging
# Checks if uvicorn is running and restarts if needed
# Intended for cron: */1 * * * * /path/to/web_server_monitor.sh

PROJECT_DIR="/mnt/e/Projects/Quantum/Video_tagging_db"
WEB_DIR="$PROJECT_DIR/web"
LOG_FILE="$PROJECT_DIR/web_server.log"
PID_FILE="$PROJECT_DIR/web_server.pid"
PORT=8000
PYTHON="/home/vano/anaconda3/envs/vllm/bin/python"
UVICORN="/home/vano/anaconda3/envs/vllm/bin/uvicorn"

log_message() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') $1" >> "$LOG_FILE"
}

check_server() {
    # Check if process exists
    if [ -f "$PID_FILE" ]; then
        PID=$(cat "$PID_FILE")
        if [ "$PID" -gt 0 ] 2>/dev/null && kill -0 "$PID" 2>/dev/null; then
            # Process exists, check if port responds
            if curl -s --max-time 5 "http://localhost:$PORT/health" > /dev/null 2>&1; then
                return 0  # Server is healthy
            fi
        fi
    fi

    # Also check by port
    if pgrep -f "uvicorn.*$PORT" > /dev/null 2>&1; then
        if curl -s --max-time 5 "http://localhost:$PORT/health" > /dev/null 2>&1; then
            # Update PID file
            pgrep -f "uvicorn.*$PORT" | head -1 > "$PID_FILE"
            return 0
        fi
    fi

    return 1  # Server not running or not healthy
}

start_server() {
    log_message "Starting web server..."

    cd "$WEB_DIR"

    # Start uvicorn in background without --reload for production
    nohup "$UVICORN" app.main:app --host 0.0.0.0 --port $PORT >> "$LOG_FILE" 2>&1 &

    NEW_PID=$!
    echo "$NEW_PID" > "$PID_FILE"

    # Wait a bit and verify
    sleep 3

    if kill -0 "$NEW_PID" 2>/dev/null; then
        log_message "Web server started with PID: $NEW_PID"
        return 0
    else
        log_message "Failed to start web server"
        return 1
    fi
}

# Main
if check_server; then
    # Server is running, nothing to do
    exit 0
else
    log_message "Web server not running or unhealthy, starting..."
    start_server
fi
