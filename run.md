cd Controller
source devel/setup.bash
roslaunch so3_quadrotor_simulator simulator_attitude_control.launch

cd Simulator
source devel/setup.bash
rosrun sensor_simulator sensor_simulator_cuda

cd YOPO
conda activate yopo
python test_yopo_ros.py --trial=1 --epoch=50

rviz -d yopo.rviz