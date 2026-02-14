# YOLO模型实时识别乒乓球

## 数据集准备

    参考数据集标注pipline的pdf

    cleanup_data.py：处理images和labels文件不对应的情况

## 模型训练和测试

    train.py:训练

    test.py:测试模型识别单张图的球

## cpp版本的环境配置

    sudo apt-get install build-essential cmake libopencv-dev



## cpp版本模型测试

1. 转换模型文件格式：

    PC端模型导出：export_onnx.py

    Jetson 端（相机主板）模型转换：/usr/src/tensorrt/bin/trtexec \
    --onnx=yolo/weights/best.onnx \
    --saveEngine=weights/best.engine \
    --fp16 \
    --minShapes=images:1x3x960x960 \
    --optShapes=images:1x3x960x960 \
    --maxShapes=images:1x3x960x960

2. 确认分支： 

    kaibot@30C-ubuntu:~/Data/Projects/TABLETENNIS_MODELPLANNER$ git status
    On branch penglin-active_vision
    Your branch is up to date with 'origin/penglin-active_vision'.

3. 启动相机：

    ./build.sh

    ./build/zed_combined_detector 

    新建一个thread,测试yolo模型推理

4. 运行模型测试单张图像：

    cd ~/Data/Projects/TABLETENNIS_MODELPLANNER/yolo

    mkdir -p build && cd build

    cd ..

    ./build/pingpong_detect


    对于不同的分辨率的图像输入，需要进行预处理才能输入模型进行预测，模型需要正方形图像输入。


5. 运行模型测试相机实时图像：

    