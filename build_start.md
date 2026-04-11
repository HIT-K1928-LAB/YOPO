# build
conda deactivate
cd ~/code/YOPO/Controller
catkin_make -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DCMAKE_BUILD_TYPE=Release

conda deactivate
cd ~/code/YOPO/Simulator
catkin_make -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DCMAKE_BUILD_TYPE=Release

# start
cd ~/code/YOPO/Controller
source devel/setup.bash
roslaunch so3_quadrotor_simulator simulator_attitude_control.launch

cd ~/code/YOPO/Simulator
source devel/setup.bash
rosrun sensor_simulator sensor_simulator_cuda

cd ~/code/YOPO/YOPO
conda activate yopo
python test_yopo_ros.py --trial=1 --epoch=50

使用tensorRT
python test_yopo_ros.py --use_tensorrt=1

cd ~/code/YOPO/YOPO
rviz -d yopo.rviz

# docker

docker run --entrypoint bash -dit \
    --runtime=nvidia --gpus all \
    --shm-size=16g \
    -e "ACCEPT_EULA=Y" --privileged \
    --network=host \
    --name zhangrun-yopo-dev \
    -e "PRIVACY_CONSENT=Y" \
    -e DISPLAY=$DISPLAY \
    -e QT_X11_NO_MITSHM=1 \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v /home/zhangrun/code:/root/code \
    base_image:ubt20-ros1-cda

xhost +
