#!/bin/bash
# Model Orchestrator v2.0 - Enhanced Autonomous Model Management
# Can be run manually or via cron for hands-free operation

# Configuration
DEFAULT_MODEL="qwen3_vl_32b.sh"
FLAG_FILE="restart_vllm.flag"
STATUS_FILE="model_status.json"
LOG_FILE="model_orchestrator.log"
LOCK_FILE="/tmp/model_orchestrator.lock"
CRON_INTERVAL=30  # seconds between checks when running in cron mode

# Current model state
CURRENT_PID=""
CURRENT_MODEL=""

# Logging with timestamps
log_message() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') $1" | tee -a "$LOG_FILE"
}

# Check if orchestrator is already running
check_orchestrator_lock() {
    if [ -f "$LOCK_FILE" ]; then
        OLD_PID=$(cat "$LOCK_FILE" 2>/dev/null)
        if [ -n "$OLD_PID" ] && kill -0 "$OLD_PID" 2>/dev/null; then
            # Another orchestrator is still running
            return 1
        else
            # Old lock file, remove it
            rm -f "$LOCK_FILE"
        fi
    fi
    
    # Create new lock
    echo $$ > "$LOCK_FILE"
    return 0
}

# Remove lock file on exit
cleanup_lock() {
    rm -f "$LOCK_FILE"
}
trap cleanup_lock EXIT

# Update model status file
update_status() {
    local status="$1"
    local model="$2" 
    local pid="$3"
    
    cat > "$STATUS_FILE" << EOF
{
  "status": "$status",
  "model": "$model",
  "pid": $pid,
  "script": "$model",
  "timestamp": "$(date '+%Y-%m-%d %H:%M:%S')",
  "uptime_seconds": $(($(date +%s) - ${START_TIME:-$(date +%s)}))
}
EOF
}

# Comprehensive model killing (like kill_all_models.sh)
kill_all_model_processes() {
    log_message "🧹 Comprehensive cleanup of ALL model processes..."
    
    # Kill current tracked process first
    if [ -n "$CURRENT_PID" ] && kill -0 "$CURRENT_PID" 2>/dev/null; then
        log_message "   Killing tracked process (PID: $CURRENT_PID)"
        kill -TERM "$CURRENT_PID" 2>/dev/null
        sleep 2
        kill -KILL "$CURRENT_PID" 2>/dev/null || true
    fi
    
    # Kill all sglang processes
    log_message "   Killing all sglang processes..."
    pkill -f "sglang" 2>/dev/null || true
    pkill -f "launch_server" 2>/dev/null || true
    pkill -f "python.*sglang" 2>/dev/null || true
    
    # Kill processes on model port
    log_message "   Killing processes on port 11434..."
    lsof -ti:11434 | xargs -r kill -9 2>/dev/null || true
    
    # Kill any python processes with model names
    log_message "   Killing Qwen model processes..."
    pkill -f "Qwen.*AWQ" 2>/dev/null || true
    pkill -f "qwen2.*vl" 2>/dev/null || true

    # Kill torch compile workers (leak ~370MB each, 12 workers = 4.4GB!)
    log_message "   Killing torch compile workers..."
    pkill -f "torch._inductor.compile_worker" 2>/dev/null || true
    pkill -f "torch.*compile_worker" 2>/dev/null || true
    
    # Kill background model shell scripts (but not this orchestrator)
    log_message "   Killing background model shell scripts..."
    ps aux | grep -E "\\.sh" | grep -v "model_orchestrator" | grep -E "(qwen|vllm)" | awk '{print $2}' | xargs -r kill -9 2>/dev/null || true
    
    # Wait for cleanup
    sleep 3
    
    # Final aggressive cleanup
    REMAINING=$(ps aux | grep -E "(sglang|launch_server)" | grep -v grep | wc -l)
    if [ "$REMAINING" -gt 0 ]; then
        log_message "   Found $REMAINING remaining processes, force killing..."
        ps aux | grep -E "(sglang|launch_server|Qwen.*AWQ)" | grep -v grep | awk '{print $2}' | xargs -r kill -9 2>/dev/null || true
        sleep 2
    fi
    
    log_message "✅ Comprehensive model cleanup completed"
    
    CURRENT_PID=""
    CURRENT_MODEL=""
    update_status "stopped" "none" 0
}

# Start a model
start_model() {
    local model_script="$1"
    
    if [ -z "$model_script" ]; then
        model_script="$DEFAULT_MODEL"
    fi
    
    # Ensure model script exists
    if [ ! -f "$model_script" ]; then
        log_message "❌ Model script not found: $model_script"
        return 1
    fi
    
    log_message "🚀 Starting model: $model_script"
    
    # Make script executable
    chmod +x "$model_script"
    
    # Start model in background and capture PID
    nohup "./$model_script" > "${model_script}.log" 2>&1 &
    CURRENT_PID=$!
    CURRENT_MODEL="$model_script"
    START_TIME=$(date +%s)
    
    log_message "   Model started with PID: $CURRENT_PID"
    
    # Wait a bit and check if process is still running
    sleep 5
    if kill -0 "$CURRENT_PID" 2>/dev/null; then
        log_message "✅ Model successfully started"
        update_status "running" "$CURRENT_MODEL" "$CURRENT_PID"
        return 0
    else
        log_message "❌ Model failed to start"
        CURRENT_PID=""
        CURRENT_MODEL=""
        update_status "failed" "none" 0
        return 1
    fi
}

# Process restart flag
process_restart_flag() {
    if [ ! -f "$FLAG_FILE" ]; then
        return 0
    fi
    
    log_message "📋 Processing restart flag..."
    
    # Read flag content (try JSON parsing, fallback to simple read)
    if command -v jq >/dev/null 2>&1; then
        MODEL_SCRIPT=$(jq -r '.model_script // "qwen2_5_vl_32b.sh"' "$FLAG_FILE" 2>/dev/null || echo "$DEFAULT_MODEL")
        ACTION=$(jq -r '.action // "restart"' "$FLAG_FILE" 2>/dev/null || echo "restart")
        REASON=$(jq -r '.reason // "manual"' "$FLAG_FILE" 2>/dev/null || echo "manual")
    else
        MODEL_SCRIPT="$DEFAULT_MODEL"
        ACTION="restart"
        REASON="manual"
    fi
    
    log_message "   Action: $ACTION"
    log_message "   Model: $MODEL_SCRIPT"
    log_message "   Reason: $REASON"
    
    # Remove flag file
    rm -f "$FLAG_FILE"
    
    # Execute action
    case "$ACTION" in
        "restart"|"switch")
            kill_all_model_processes
            sleep 3
            start_model "$MODEL_SCRIPT"
            ;;
        "stop")
            kill_all_model_processes
            log_message "🛑 Model stopped by request"
            ;;
        "start")
            if [ -z "$CURRENT_PID" ] || ! kill -0 "$CURRENT_PID" 2>/dev/null; then
                start_model "$MODEL_SCRIPT"
            else
                log_message "⚠️ Model already running (PID: $CURRENT_PID)"
            fi
            ;;
        *)
            log_message "❓ Unknown action: $ACTION"
            ;;
    esac
}

# Check model health
check_model_health() {
    if [ -n "$CURRENT_PID" ] && [ "$CURRENT_PID" -gt 0 ] 2>/dev/null && kill -0 "$CURRENT_PID" 2>/dev/null; then
        # Process is running, check if API is responding
        if curl -s -m 5 http://127.0.0.1:11434/v1/models >/dev/null 2>&1; then
            return 0  # Healthy
        else
            # Grace period: don't restart if model started < 5 minutes ago
            if [ -f "$STATUS_FILE" ]; then
                MODEL_TIMESTAMP=$(python3 -c "import json; print(json.load(open('$STATUS_FILE')).get('timestamp', ''))" 2>/dev/null || echo "")
                if [ -n "$MODEL_TIMESTAMP" ]; then
                    START_EPOCH=$(date -d "$MODEL_TIMESTAMP" +%s 2>/dev/null)
                    NOW_EPOCH=$(date +%s)
                    ELAPSED=$((NOW_EPOCH - START_EPOCH))
                    if [ "$ELAPSED" -lt 300 ]; then
                        log_message "⏳ Model loading (${ELAPSED}s < 300s grace period), API not ready yet"
                        return 0  # Consider healthy during grace period
                    fi
                fi
            fi
            log_message "⚠️ Model process running but API not responding (grace period expired)"
            return 1  # Unhealthy
        fi
    else
        # Process not running
        if [ -n "$CURRENT_PID" ]; then
            log_message "⚠️ Tracked model process (PID: $CURRENT_PID) is not running"
            CURRENT_PID=""
            CURRENT_MODEL=""
            update_status "stopped" "none" 0
        fi
        return 1  # Not running
    fi
}

# Auto-restart unhealthy models
auto_restart_if_needed() {
    if ! check_model_health; then
        if [ -n "$CURRENT_MODEL" ]; then
            log_message "🔄 Auto-restarting unhealthy model: $CURRENT_MODEL"
            kill_all_model_processes
            sleep 3
            start_model "$CURRENT_MODEL"
        else
            log_message "🚀 No model running, starting default model"
            start_model "$DEFAULT_MODEL"
        fi
    fi
}

# Print usage
usage() {
    echo "Model Orchestrator v2.0 - Enhanced Autonomous Model Management"
    echo ""
    echo "Usage: $0 [mode]"
    echo ""
    echo "Modes:"
    echo "  daemon    - Run as daemon (default, continuous monitoring)"
    echo "  cron      - Run once for cron job (single check and exit)"
    echo "  status    - Show current model status"
    echo "  stop      - Stop all models"
    echo "  restart   - Restart current model"
    echo "  cleanup   - Emergency cleanup of all model processes"
    echo ""
    echo "Examples:"
    echo "  $0 daemon           # Run continuously"
    echo "  $0 cron             # Single check (for cron)"
    echo "  */1 * * * * $0 cron # Add to crontab for every minute"
}

# Main execution modes
main() {
    local mode="${1:-daemon}"
    
    case "$mode" in
        "daemon")
            # Check lock to prevent multiple instances
            if ! check_orchestrator_lock; then
                echo "❌ Another orchestrator instance is already running"
                exit 1
            fi
            
            log_message "🎯 Model Orchestrator v2.0 starting in DAEMON mode"
            
            # Start default model if nothing is running
            auto_restart_if_needed
            
            # Main daemon loop
            while true; do
                # Process any restart flags
                process_restart_flag
                
                # Check model health every few cycles
                sleep 30
                auto_restart_if_needed
                
                sleep 5
            done
            ;;
            
        "cron")
            # Single check mode for cron jobs
            log_message "🔍 Model Orchestrator v2.0 CRON check"

            # Check lock to prevent conflicts with daemon mode
            if [ -f "$LOCK_FILE" ]; then
                OLD_PID=$(cat "$LOCK_FILE" 2>/dev/null)
                if [ -n "$OLD_PID" ] && kill -0 "$OLD_PID" 2>/dev/null; then
                    # Daemon is running, just process flags
                    process_restart_flag
                    exit 0
                fi
            fi

            # Load current state from status file (critical for cron!)
            if [ -f "$STATUS_FILE" ]; then
                CURRENT_PID=$(python3 -c "import json; print(json.load(open('$STATUS_FILE')).get('pid', 0))" 2>/dev/null || echo "0")
                CURRENT_MODEL=$(python3 -c "import json; print(json.load(open('$STATUS_FILE')).get('model', 'none'))" 2>/dev/null || echo "none")
                log_message "   Loaded state: PID=$CURRENT_PID, MODEL=$CURRENT_MODEL"
            fi

            # No daemon running, do single check
            process_restart_flag
            auto_restart_if_needed
            ;;
            
        "status")
            if [ -f "$STATUS_FILE" ]; then
                echo "📊 Current model status:"
                cat "$STATUS_FILE" | jq . 2>/dev/null || cat "$STATUS_FILE"
            else
                echo "❌ No status file found"
                exit 1
            fi
            ;;
            
        "stop")
            log_message "🛑 Manual stop requested"
            kill_all_model_processes
            echo "✅ All models stopped"
            ;;
            
        "restart")
            log_message "🔄 Manual restart requested"
            kill_all_model_processes
            sleep 3
            start_model "$DEFAULT_MODEL"
            ;;
            
        "cleanup")
            log_message "🧹 Manual cleanup requested"
            kill_all_model_processes
            echo "✅ Emergency cleanup completed"
            ;;
            
        "help"|"-h"|"--help")
            usage
            ;;
            
        *)
            echo "❌ Unknown mode: $mode"
            usage
            exit 1
            ;;
    esac
}

# Run main function
main "$@"