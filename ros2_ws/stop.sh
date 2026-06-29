#!/bin/bash
echo "Stopping all sim processes..."
for p in gz-server gz-client gz-sim rviz2 move_group robot_state_publisher parameter_bridge scene_publisher srdf_publisher; do
    pkill -f "$p" 2>/dev/null
done
sleep 1
REMAINING=$(ps aux | grep -E "gz sim|gazebo|move_group|rviz2|robot_state|parameter_bridge|scene_publisher|srdf_publisher" | grep -v grep | wc -l)
if [ "$REMAINING" -gt 0 ]; then
    echo "Force kill $REMAINING remaining processes..."
    ps aux | grep -E "gz sim|gazebo|move_group|rviz2|robot_state|parameter_bridge|scene_publisher|srdf_publisher" | grep -v grep | awk '{print $2}' | xargs kill -9 2>/dev/null
fi
tmux kill-session -t handeye_sim 2>/dev/null || true
# Gazebo 模型清理
gz model -m calibration_plate --remove 2>/dev/null || true
gz model -m fanuc_robot --remove 2>/dev/null || true
echo "All stopped"
