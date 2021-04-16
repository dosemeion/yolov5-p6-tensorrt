# yolov5-p6-tensorrt

The TensorRT implementation is base on [tensorrtx](https://github.com/wang-xinyu/tensorrtx)


## Different versions of yolov5

Currently, It support yolov5-p6 v4.0.

- For yolov5 v4.0, download .pt from [yolov5 release v4.0](https://github.com/ultralytics/yolov5/releases/tag/v4.0), `git clone -b v4.0 https://github.com/ultralytics/yolov5.git` and `git clone https://github.com/wang-xinyu/tensorrtx.git`.

## Config

- Choose the model s/m/l/x by `NET` macro in yolov5-p6.cpp
- Input shape defined in yololayer.h
- Number of classes defined in yololayer.h, **DO NOT FORGET TO ADAPT THIS, If using your own model**
- INT8/FP16/FP32 can be selected by the macro in yolov5-p6.cpp, **INT8 need more steps, pls follow `How to Run` first and then go the `INT8 Quantization` below**
- GPU id can be selected by the macro in yolov5.cpp
- NMS thresh in yolov5-p6.cpp
- BBox confidence thresh in yolov5-p6.cpp
- Batch size in yolov5-p6.cpp

## How to Run, yolov5s as example

1. generate .wts from pytorch with .pt, or download .wts from model zoo

```
// git clone src code according to `Different versions of yolov5` above
// download https://github.com/ultralytics/yolov5/releases/download/v4.0/yolov5s6.pt
// copy tensorrtx/yolov5/gen_wts.py into ultralytics/yolov5
// ensure the file name is yolov5s6.pt and yolov5s.wts in gen_wts.py
// go to ultralytics/yolov5
python gen_wts.py
// a file 'yolov5s6.wts' will be generated.
```

2. build tensorrtx/yolov5 and run

```
// put yolov5s6.wts into yolov5-p6-tensorrt
// go to yolov5-p6-tensorrt
// update CLASS_NUM in yololayer.h if your model is trained on custom dataset
mkdir build
cd build
cmake ..
make
sudo ./yolov5-p6 -s [.wts] [.engine] [s/m/l/x or c gd gw]  // serialize model to plan file
sudo ./yolov5-p6 -d [.engine] [image folder]  // deserialize and run inference, the images in [image folder] will be processed.
// For example yolov5s6
sudo ./yolov5-p6 -s yolov5s6.wts yolov5s6.engine s
sudo ./yolov5-p6 -d yolov5s.engine ../samples
// For example Custom model with depth_multiple=0.17, width_multiple=0.25 in yolov5.yaml
sudo ./yolov5-p6 -s yolov5_custom.wts yolov5.engine c 0.17 0.25
sudo ./yolov5-p6 -d yolov5.engine ../samples
```

3. check the images generated, as follows. _zidane.jpg and _bus.jpg

4. optional, load and run the tensorrt model in python

```
// install python-tensorrt, pycuda, etc.
// ensure the yolov5s6.engine and libmyplugins.so have been built
python yolov5_trt.py
```

## More infomation

please see the readme in (https://github.com/wang-xinyu/tensorrtx)

