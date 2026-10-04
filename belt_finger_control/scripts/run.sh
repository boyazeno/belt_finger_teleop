#!/usr/bin/env bash
set -e  # Exit immediately on any error
trap "trap - SIGTERM && echo 'Shutdown initiated, killing all processes...' && kill -- -$$" SIGINT SIGTERM EXIT  

echo "Starting roscore in background..."
roscore & 
ROS_PID=$!  

sleep 2

# echo "Launching gripper.py..."
# python gripper.py & 
# GRIPPER_PID=$!  


PIPE=/tmp/input_pipe
rm -f "$PIPE"
mkfifo "$PIPE"                       

echo "Launching teleop_conveyer_hand_node.py..."
python teleop_conveyer_hand_node.py < "$PIPE" & 
TELEOP_PID=$!

echo "Type a command and press Enter to send it to teleop_conveyer_hand_node.py:"
while read -r line; do
  echo "$line" > "$PIPE"          # Forwarded to input() in the Python node
done

wait $TELEOP_PID  
kill $ROS_PID