# tensorflow-keras-yolov3
(cocoapi mAP计算在下方↓↓↓)
[![license](https://img.shields.io/github/license/mashape/apistatus.svg)](LICENSE)

---

### Quick Start

1. The test environment is
    - cudatoolkit  9.2
    - cudnn 7.2.1
    - Python 3.6.8
    - Keras 2.2.0
    - tensorflow 1.10.0
    - pillow = 5.4.1
    - matplotlib 3.0.2

2. Download YOLOv3 weights from [YOLO website](http://pjreddie.com/darknet/yolo/).
3. Convert the Darknet YOLO model to a Keras model .h5 file. 
4. Modified default converted model path in yolo.py line26 (default in '/home/common/pretrained_models/yolo.h5')

### Run single image detection demo
```
wget https://pjreddie.com/media/files/yolov3.weights
python convert.py yolov3.cfg yolov3.weights /home/common/pretrained_models/yolo.h5
python yolo_video.py [OPTIONS...] --image, for image detection mode, OR
python yolo_video.py [video_path] [output_path (optional)]
For Tiny YOLOv3, just do in a similar way, just specify model path and anchor path with `--model model_file` and `--anchors anchor_file`.
```
---
### Calcualte mAP on cocoapi
```
1. cd tensorflow-keras-yolov3
2. pip install cython # solution of issue:(gcc: error: pycocotools/_mask.c: No such file or directory)
3. sudo rm -rf cocoapi && git clone https://github.com/cocodataset/cocoapi && cd cocoapi/PythonAPI && make && cd ../.. && cp -r cocoapi/PythonAPI/pycocotools ./
4. Use `python yolo_valid.py` to test the official YOLOv3 weights.
```

---
### Other usage
Use --help to see usage of yolo_video.py:
```
usage: yolo_video.py [-h] [--model MODEL] [--anchors ANCHORS]
                     [--classes CLASSES] [--gpu_num GPU_NUM] [--image]
                     [--input] [--output]

positional arguments:
  --input        Video input path
  --output       Video output path

optional arguments:
  -h, --help         show this help message and exit
  --model MODEL      path to model weight file, default model_data/yolo.h5
  --anchors ANCHORS  path to anchor definitions, default
                     model_data/yolo_anchors.txt
  --classes CLASSES  path to class definitions, default
                     model_data/coco_classes.txt
  --gpu_num GPU_NUM  Number of GPU to use, default 1
  --image            Image detection mode, will ignore all positional arguments
```
4. MultiGPU usage: use `--gpu_num N` to use N GPUs. It is passed to the [Keras multi_gpu_model()](https://keras.io/utils/#multi_gpu_model).
---
### Training

1. Generate your own annotation file and class names file.  
    One row for one image;  
    Row format: `image_file_path box1 box2 ... boxN`;  
    Box format: `x_min,y_min,x_max,y_max,class_id` (no space).  
    For VOC dataset, try `python voc_annotation.py`  
    Here is an example:
    ```
    path/to/img1.jpg 50,100,150,200,0 30,50,200,120,3
    path/to/img2.jpg 120,300,250,600,2
    ...
    ```

2. Make sure you have run `python convert.py -w yolov3.cfg yolov3.weights model_data/yolo_weights.h5`  
    The file model_data/yolo_weights.h5 is used to load pretrained weights.

3. Modify train.py and start training.  
    `python train.py`  
    Use your trained weights or checkpoint weights with command line option `--model model_file` when using yolo_video.py
    Remember to modify class path or anchor path, with `--classes class_file` and `--anchors anchor_file`.

If you want to use original pretrained weights for YOLOv3:  
    1. `wget https://pjreddie.com/media/files/darknet53.conv.74`  
    2. rename it as darknet53.weights  
    3. `python convert.py -w darknet53.cfg darknet53.weights model_data/darknet53_weights.h5`  
    4. use model_data/darknet53_weights.h5 in train.py

---
### Some issues to know



1. Default anchors are used. If you use your own anchors, probably some changes are needed.

2. The inference result is not totally the same as Darknet but the difference is small.

3. The speed is slower than Darknet. Replacing PIL with opencv may help a little.

4. Always load pretrained weights and freeze layers in the first stage of training. Or try Darknet training. It's OK if there is a mismatch warning.


# tensorflow-keras-yolov3
