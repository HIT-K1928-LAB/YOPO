#!/bin/bash

SESH="yopo_tmux_session"
ROS_HOSTNAME="127.0.0.1"
ROS_MASTER_URI="http://127.0.0.1:11311"

# 当前脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

tmux has-session -t $SESH 2>/dev/null

if [ $? != 0 ]; then
    tmux new-session -d -s $SESH -n "Controller"

    tmux send-keys -t $SESH:Controller "export ROS_MASTER_URI=$ROS_MASTER_URI" C-m
    tmux send-keys -t $SESH:Controller "export ROS_HOSTNAME=$ROS_HOSTNAME" C-m
    tmux send-keys -t $SESH:Controller "cd \"$SCRIPT_DIR\"/Controller" C-m
    tmux send-keys -t $SESH:Controller "source \"$SCRIPT_DIR\"/Controller/devel/setup.bash" C-m
    tmux send-keys -t $SESH:Controller "roslaunch so3_quadrotor_simulator simulator_attitude_control.launch"

    tmux new-window -t $SESH -n "Simulator"
    tmux send-keys -t $SESH:Simulator "export ROS_MASTER_URI=$ROS_MASTER_URI" C-m
    tmux send-keys -t $SESH:Simulator "export ROS_HOSTNAME=$ROS_HOSTNAME" C-m
    tmux send-keys -t $SESH:Simulator "cd \"$SCRIPT_DIR\"/Simulator" C-m
    tmux send-keys -t $SESH:Simulator "source \"$SCRIPT_DIR\"/Simulator/devel/setup.bash" C-m
    tmux send-keys -t $SESH:Simulator "rosrun sensor_simulator sensor_simulator_cuda "

    tmux new-window -t $SESH -n "yopo"
    tmux send-keys -t $SESH:yopo "export ROS_MASTER_URI=$ROS_MASTER_URI" C-m
    tmux send-keys -t $SESH:yopo "export ROS_HOSTNAME=$ROS_HOSTNAME" C-m
    tmux send-keys -t $SESH:yopo "cd \"$SCRIPT_DIR\"/YOPO" C-m
    tmux send-keys -t $SESH:yopo "conda activate yopo" C-m
    tmux send-keys -t $SESH:yopo "python test_yopo_ros.py --use_tensorrt=1"

    tmux new-window -t $SESH -n "rviz"
    tmux send-keys -t $SESH:rviz "export ROS_MASTER_URI=$ROS_MASTER_URI" C-m
    tmux send-keys -t $SESH:rviz "export ROS_HOSTNAME=$ROS_HOSTNAME" C-m
    tmux send-keys -t $SESH:rviz "cd \"$SCRIPT_DIR\"/YOPO" C-m
    tmux send-keys -t $SESH:rviz "rviz -d yopo.rviz"

    tmux select-window -t $SESH:Controller
fi

tmux attach-session -t $SESH