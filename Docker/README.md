# YOPO环境搭建

## PC Docker 环境构建

``` bash
# pc
make yopo_pc

```

可以直接进行YOPO的仿真测试

## orin Docker 环境构建

``` bash
# orin

make yopo_orin

```

以下部分和在PC上进行仿真测试略有不同

### PC和orin架构不同，pip install requirements报错

``` bash
(yopo) root@ubuntu:~/code/YOPO/YOPO# pip install -r requirements.txt
Looking in indexes: https://pypi.org/simple, https://download.pytorch.org/whl/cu118
ERROR: Could not find a version that satisfies the requirement torch==2.4.1+cu118 (from versions: 1.8.0, 1.8.1, 1.9.0, 1.10.0, 1.10.1, 1.10.2, 1.11.0, 1.12.0, 1.12.1, 1.13.0, 1.13.1, 2.0.0, 2.0.1, 2.1.0, 2.1.1, 2.1.2, 2.2.0, 2.2.1, 2.2.2, 2.3.0, 2.3.1, 2.4.0, 2.4.1)
ERROR: No matching distribution found for torch==2.4.1+cu118
```

当前系统：
- L4T: R35.6.3
- JetPack: 5.15
- CUDA: 11.4
- 架构: aarch64

#### 解决

[版本对应关系](https://blog.csdn.net/shiwanghualuo/article/details/122860521)

| torch       | 2.1.0  |
| :---------- | :----- |
| torchvision | 0.16.0 |
| torchaudio  | 2.1.0  |

以下均在conda虚拟环境中运行安装

``` bash
# 下载torch-2.1.0a0+41361538.nv23.06-cp38-cp38-linux_aarch64.whl
sudo apt update
sudo apt install -y libopenblas-dev libopenmpi-dev libomp-dev

pip install --no-cache https://developer.download.nvidia.com/compute/redist/jp/v512/pytorch/torch-2.1.0a0+41361538.nv23.06-cp38-cp38-linux_aarch64.whl
```

``` python
# 测试torch
>>> import torch
>>> print(torch.__version__)
2.0.0a0+8aa34602.nv23.03
>>> print(torch.cuda.is_available())
True
>>> print(torch.version.cuda)
11.4
>>> print(torch.cuda.get_device_name(0))
Orin
```

``` bash
# 安装torchvision
sudo apt install -y \  
libjpeg-dev \  
zlib1g-dev \  
libpng-dev \  
libopenblas-dev \  
libavcodec-dev \  
libavformat-dev \  
libswscale-dev \  
libomp-dev

git clone --branch release/0.16 https://github.com/pytorch/vision.git
cd vision

conda activate yopo
python3 setup.py install
```

```python
# 测试torchvision
import torchvision  
print(torchvision.__version__)
```

``` bash
#安装torchaudio
git clone --branch release/2.1.0 https://github.com/pytorch/audio.git
cd audio

# 需要额外安装高CMake版本
wget https://github.com/Kitware/CMake/releases/download/v3.27.9/cmake-3.27.9-linux-aarch64.sh
chmod +x cmake-3.27.9-linux-aarch64.sh
sudo mkdir -p /opt/cmake-3.27
sudo chmod 755 /opt/cmake-3.27
sudo ./cmake-3.27.9-linux-aarch64.sh --skip-license --prefix=/opt/cmake-3.27

# 此时系统有两个版本cmake
# /usr/bin/cmake # 3.16.3
# /opt/cmake-3.27/bin/cmake

apt update
apt install -y  build-essential ninja-build gcc g++ make pkg-config
export PATH=/opt/cmake-3.27/bin:$PATH
python3 setup.py install
```

``` python
# 测试torchaudio
import torchaudio
print(torchaudio.__version__)
```

修改requirements.txt文件，把不需要安装的注释掉
``` txt
--extra-index-url https://download.pytorch.org/whl/cu118

# torch==2.4.1+cu118
# torchaudio==2.4.1+cu118
# torchvision==0.19.1+cu118
# opencv-python==4.11.0.86
scipy==1.10.1
scikit-build==0.18.1
ruamel-yaml==0.17.21
numpy==1.22.3
tensorboard==2.14.0
# open3d==0.19.0
rich==14.0.0
empy==4.2
catkin_pkg
rospkg
netifaces
```

然后进行安装

``` bash
pip install -r requirements.txt
```

### orin板运行yopo时提示no module named cv2

使用 NVIDIA 预装的 OpenCV（性能最强）
NVIDIA 的 JetPack 系统镜像里其实自带了一个针对 Orin 硬件优化过（支持 GStreamer 和 CUDA 加速）的 OpenCV。但默认情况下，虚拟环境访问不到它。

如果你想在虚拟环境里使用系统自带的 OpenCV，可以手动创建一个软连接（假设你使用的是 Python 3.8）：
1. 找到系统自带 OpenCV 的路径（通常在下面这个位置）
ls /usr/lib/python3.8/dist-packages/cv2*
2. 将它链接到你的虚拟环境路径下（请将路径替换为你虚拟环境的实际路径）
cd ~/code/YOPO/YOPO/venv/lib/python3.8/site-packages/  # 进入你的虚拟环境库目录
ln -s /usr/lib/python3.8/dist-packages/cv2/python-3.8/cv2.cpython-38-aarch64-linux-gnu.so cv2.so

之后可能报错
``` bash
>>> import cv2
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
ImportError: /lib/aarch64-linux-gnu/libp11-kit.so.0: undefined symbol: ffi_type_pointer, version LIBFFI_BASE_7.0
```

虚拟环境中运行
``` bash
conda install -c conda-forge libffi=3.3
```

### orin跑仿真算力不够

目前在orin上运行仿真地图、rviz，由于rviz算力不够，导致运算速度慢表现不好，如何过主从机的方式，在主机侧运行仿真，orin只运行推理的python文件

### 通过无线wifi构建ROS Master-Slave 分布式部署模式

主机的.bashrc

``` bash
export ROS_MASTER_URI=http://192.168.31.6:11311
export ROS_HOSTNAME=192.168.31.6
```

orin的.bashrc

``` bash
export ROS_MASTER_URI=http://192.168.31.6:11311
export ROS_HOSTNAME=192.168.31.97
```

主机测试发布

``` bash
rostopic pub /test std_msgs/String "data: 'hello'"
```

orin测试接收

``` bash
rostopic echo /test
```

---


