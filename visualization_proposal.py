from mmdet.apis import init_detector
from mmdet.apis import inference_detector
from mmdet.models.task_modules.coders.delta_xywh_bbox_coder import delta2bbox
import torch
from torchvision.ops import nms
import cv2
import os

def hook_fn(module, input, output):

    output_s1 = output
    
    global stored_output
    stored_output = output_s1

config_file = 'configs/faster_rcnn_r50_fpn_1x_voc.py'
checkpoint_file = 'save/faster_rcnn_epoch_16.pth'


# img = 'test_imgs/2008_000002.jpg'
path = './test_proposals'
os.makedirs('output_proposals',exist_ok=True)

model = init_detector(config_file, checkpoint_file, device='cuda:0')


hook = model.rpn_head.register_forward_hook(hook_fn)

for f in os.listdir(path):

    img = os.path.join(path,f)
    results = inference_detector(model, img)

    rpn_cls_output,rpn_reg_output = stored_output

    img_shape = [feature.shape[2:] for feature in rpn_cls_output]
    anchors =  model.rpn_head.anchor_generator.grid_anchors(img_shape)

    layers = len(rpn_cls_output)
    proposals = []
    scores = []
    for i in range(layers):
        proposal = delta2bbox(anchors[i], rpn_reg_output[i].view(-1,4), means=[0., 0., 0., 0.], stds=[1., 1., 1., 1.])
        score = torch.sigmoid(rpn_cls_output[i].view(-1,1))
        proposals.append(proposal)
        scores.append(score)
    proposals = torch.cat(proposals,dim=0)
    scores = torch.cat(scores,dim=0)
    scores = scores.view(-1)

    keep = nms(proposals, scores, 0.7)

    proposals = proposals[keep]
    scores = scores[keep]

    topk_scores, topk_inds = scores.topk(100)
    proposals = proposals[topk_inds]
    scores = scores[topk_inds]

    # 绘制候选框
    img_array = cv2.imread(img)
    for det in proposals:
        bbox = det[:4]
        cv2.rectangle(img_array, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 0, 255), 1)

    # cv2.imshow("Proposals", img_array)
    cv2.imwrite(f'./output_proposals/{f[:-4]}_proposals.{f[-3:]}', img_array)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

hook.remove()
