# Preparation
## environment
```
bash mmdet.sh
```
Make sure the version of mmdet is >=3.2.0.
## dataset
The data is not in this repo, to download it you can go [`VOC2012`](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar)
It is recommaned to put the data file in the dataset folder like
```
data/
    VOCdevkit/
        VOC2012/
            JPEGImages/
            ImageSets/
            Annotations/
```

# Train & Evaluation
Because of the mmdetection framework, if you want to change the settings, you need to go to the corresponding config file and edit it.
Here are some key config files
```
configs/faster_rcnn_r50_fpn_1x_voc.py
configs/yolov3_d53_8xb8-ms-608-273e_voc.py
configs/_base_/models/faster-rcnn_r50_fpn.py
configs/_base_/datasets/voc2012.py
configs/_base_/schedules/schedule_1x.py 
configs/_base_/default_runtime.py
```

## for train
```
#  train for Faster R-CNN
python train.py configs/faster_rcnn_r50_fpn_1x_voc.py
#  train for YOLO V3
python train.py configs/yolov3_d53_8xb8-ms-608-273e_voc.py
```

## for evaluation
```
#  test for Faster R-CNN
python test.py configs/faster_rcnn_r50_fpn_1x_voc.py save/faster_rcnn.pth
#  test for YOLO V3
python test.py configs/yolov3_d53_8xb8-ms-608-273e_voc.py save/yolo.pth
```
The weights after model training can be downloaded [`here`](https://drive.google.com/drive/folders/15a8OrOcwF9sMXn3jfRV5TihvW1-qftTE?usp=drive_link)

## for proposals visualization in the first stage of Faster R-CNN
Remember to check the paths in the code. Simply put the test images in the `test_proposals` folder, the results will be saved in the `output_proposals` folder
```
python visualization_proposal.py
```

## for External data test
Again remember to check the paths in the code. Simply put the test images in the `test_examples` folder, the results will be saved in the `output_examples` folder
```
# Both Faster R-CNN and YOLO V3 results will be generated.
python test_image.py
```




