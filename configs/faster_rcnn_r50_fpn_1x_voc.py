# 在configs/faster_rcnn/faster_rcnn_r50_fpn_1x_voc_custom.py中

_base_ = [
    './_base_/models/faster-rcnn_r50_fpn.py',
    './_base_/datasets/voc2012.py',
    './_base_/schedules/schedule_1x.py', 
    './_base_/default_runtime.py'
]
model = dict(
    roi_head=dict(
        bbox_head=dict(num_classes=20)))



evaluation = dict(interval=1, metric='bbox')
# custom_hooks = dict(type='CheckInvalidLossHook', interval=1, priority='VERY_LOW')
# custom_hooks = [dict(type='CheckInvalidLossHook', interval=1)]
visualizer = dict(vis_backends = [dict(type='TensorboardVisBackend')])
