xhost +
docker run -it --name yopo_dev --gpus all --runtime=nvidia --net=host --privileged -e DISPLAY=$DISPLAY -e QT_X11_NO_MITSHM=1 -v /tmp/.X11-unix:/tmp/.X11-unix -v /home/zhangrun/code:/root/code ubuntu/cuda/ros/miniconda/gcc:20.04-11.4-noetic-25.11.1-8 bash

catkin_make -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DCMAKE_BUILD_TYPE=Release