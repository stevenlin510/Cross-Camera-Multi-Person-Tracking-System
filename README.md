# Cross-Camera-Multi-Person-Tracking

## Introduction 

* Implementation by **Wei-Cheng Lin**, Service Systems Technology Center, Industrial Technology Research Institute, Taiwan. 
* Develop a client-server person re-identification system to track people under 4 ipcams in a single room. In our setting, we have overall 3 rooms and 12 ipcams. 

## Operating System 
- [x] Windows 10
- [x] Ubuntu 18.04

## Install

```shell 
conda create -n reid python=3.7
conda activate reid
pip install -r requirements.txt
```
Test under **Pytorch 1.7 & Cuda 11.0**, please intall them from their offical website.

Clone the [TorchReid](https://github.com/KaiyangZhou/deep-person-reid.git) repository and build it from source. 
```shell 
git clone https://github.com/KaiyangZhou/deep-person-reid.git
cd deep-person-reid
python setup.py develop
```

## Configuration
Download the model weights (you prefer) and put them into `weight` folder.

[`yolov7.pt`](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7.pt) [`yolov7x.pt`](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7x.pt) [`yolov7-w6.pt`](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-w6.pt) [`yolov7-e6.pt`](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-e6.pt) [`yolov7-d6.pt`](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-d6.pt) [`yolov7-w6-person.pt`](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-w6-person.pt)

[`osnet_ain_x1_0`](https://drive.google.com/file/d/1SigwBE6mPdqiJMqhuIY4aqC7--5CsMal/view) [`osnet_x1_0`](https://drive.google.com/file/d/1IosIFlLiulGIjwW3H8uMRmx3MzPwf86x/view)

*Rename the reid model weight into `osnet_ain_x1_0.pth` and `osnet_x1_0.pth`, respectively*

Head to `config.py` file, and modify the ipcam's address and additional setup as you want. 

We use **four** ipcams as our default setup.

## Inference 

* Step 1 : In the client computer, run 
```shell
python client.py
```

* Step 2 : In the server computer, run
```shell
python server.py
```
**Be sure to run the `client.py` file before runing the `server.py` file.**

## Acknowledgement

Big thanks to [`Multi-Camera-Multi-Person-Tracking`](https://github.com/naufalzhafran/Multi-Camera-Multi-Person-Tracking) for sharing code.

* [`Multi-Camera-Multi-Person-Tracking`](https://github.com/naufalzhafran/Multi-Camera-Multi-Person-Tracking)
* [`TorchReid`](https://github.com/KaiyangZhou/deep-person-reid)
* [`Yolov7`](https://github.com/WongKinYiu/yolov7)
