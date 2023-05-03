# Object Detection using YOLO

```sh
cd src
```

yolo on images
```sh
py yolo.py --image-path='/path/to/image/'
```

yolo on video
```sh
py yolo.py --video-path='/path/to/video/'
```

yolo on webcam
```sh
py yolo.py
```

```sh
py yolo.py --image-path='./images/1.jpg' --labels='./cfg/yolo-coco/coco.names' --weights='././cfg/yolo-coco/yolov3.weights' --config='./cfg/yolo-coco/yolov3.cfg'
```

```sh
py yolo.py --video-path='./mangalore-images/2.mp4' --labels='./cfg/yolo-coco/coco.names' --weights='././cfg/yolo-coco/yolov3.weights' --config='./cfg/yolo-coco/yolov3.cfg'
```