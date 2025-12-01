#!/bin/bash
# Emergency script to kill all VLLM/sglang processes
# Use when orchestrator cleanup fails

echo "🚨 EMERGENCY CLEANUP: Killing ALL VLLM/sglang processes..."

# Kill orchestrator first if running
echo "1. Killing model orchestrator..."
pkill -f "model_orchestrator" || true

# Kill all sglang related processes
echo "2. Killing all sglang processes..."
pkill -f "sglang" -9 || true
pkill -f "launch_server" -9 || true

# Kill specific process patterns
echo "3. Killing Qwen model processes..."
pkill -f "Qwen.*AWQ" -9 || true
pkill -f "Qwen2.5-VL-32B-Instruct-AWQ" -9 || true

# Kill processes on port 11434
echo "4. Killing processes on port 11434..."
lsof -ti:11434 | xargs -r kill -9 || true

# Kill background shell scripts
echo "5. Killing background model shell scripts..."
pkill -f "qwen2_5_vl_32b.sh" || true

# Show remaining processes
echo "6. Checking for remaining processes..."
REMAINING=$(ps aux | grep -E "(sglang|launch_server|Qwen)" | grep -v grep)
if [ -n "$REMAINING" ]; then
    echo "❌ Found remaining processes:"
    echo "$REMAINING"
    
    # Force kill by PID
    echo "7. Force killing remaining processes by PID..."
    echo "$REMAINING" | awk '{print $2}' | xargs -r kill -9 || true
    
    sleep 2
    
    # Final check
    FINAL_CHECK=$(ps aux | grep -E "(sglang|launch_server|Qwen)" | grep -v grep)
    if [ -n "$FINAL_CHECK" ]; then
        echo "❌ Still found processes after force kill:"
        echo "$FINAL_CHECK"
    else
        echo "✅ All processes successfully killed"
    fi
else
    echo "✅ No remaining processes found"
fi

# Check GPU memory
echo "8. Checking GPU memory usage..."
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits
else
    echo "   nvidia-smi not available"
fi

# Clean up temp files
echo "9. Cleaning up temp files..."
rm -f restart_vllm.flag model_status.json 2>/dev/null || true

echo "🧹 Emergency cleanup completed!"
echo ""
echo "To restart the system:"
echo "1. ./model_orchestrator.sh (in one terminal)"
echo "2. python phase2_vllm_analysis.py (in another terminal)"