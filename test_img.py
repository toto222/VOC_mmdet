import cv2
from mmdet.apis import init_detector, inference_detector
import random
import os

path = './test_examples'
os.makedirs('output_examples',exist_ok=True)

model_names = ['yolo','faster_rcnn']
for model_name in model_names:

    if model_name =='yolo':
        config_file = 'configs/yolov3_d53_8xb8-ms-608-273e_voc.py'
        checkpoint_file = 'save/yolo.pth'
    else:
        config_file = 'configs/faster_rcnn_r50_fpn_1x_voc.py'
        checkpoint_file = 'save/faster_rcnn.pth'



    model = init_detector(config_file, checkpoint_file, device='cuda:0')

    filelis = os.listdir(path)

    for f in filelis:
        img_path = os.path.join(path,f)
        result = inference_detector(model, img_path)


        class_names = [
            "aeroplane",
            "bicycle",
            "bird",
            "boat",
            "bottle",
            "bus",
            "car",
            "cat",
            "chair",
            "cow",
            "diningtable",
            "dog",
            "horse",
            "motorbike",
            "person",
            "pottedplant",
            "sheep",
            "sofa",
            "train",
            "tvmonitor"
        ]

        img = cv2.imread(img_path)
        # draw = ImageDraw.Draw(img)

        bbox_result, segm_result = result, None
        # import pdb;pdb.set_trace()
        bboxes = bbox_result.pred_instances.bboxes
        labels = result.pred_instances.labels
        scores = result.pred_instances.scores
        inds = scores > 0.6
        bboxes = bboxes[inds, :]
        labels = labels[inds]
        scores = scores[inds]
        # font = ImageFont.truetype(size=20)
        def random_color():
            b = random.randint(0,255)
            g = random.randint(0,255)
            r = random.randint(0,255)
        
            return (b,g,r)

        for i, det in enumerate(bboxes):
            
            bbox = det[:4]
            bbox = [b.item() for b in bbox]
            score = scores[i]
            label = labels[i]
            x1, y1, x2, y2 = bbox
            name = f'{class_names[label]}:{score:.2f}'
            cv2.rectangle(img,(int(x1),int(y1)),(int(x2),int(y2)),random_color(),thickness=5)
            cv2.putText(img, text=name, org=(int(x1)+4, int(y1)+12), fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
                fontScale=1, thickness=2, lineType=cv2.LINE_AA, color=(0, 0, 255))
        
            # cv2.imshow('result',src_img)
            # cv2.waitKey()
            # cv2.destroyAllWindows()

        cv2.imwrite(f'output_examples/{f[:-4]}_{model_name}.{f[-3:]}',img)

