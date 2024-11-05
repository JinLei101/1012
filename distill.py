import warnings
warnings.filterwarnings('ignore')
import argparse, yaml, copy
from ultralytics.models.yolo.detect.distill import DetectionDistiller
# from ultralytics.models.yolo.segment.distill import SegmentationDistiller
# from ultralytics.models.yolo.pose.distill import PoseDistiller
# from ultralytics.models.yolo.obb.distill import OBBDistiller

if __name__ == '__main__':
    param_dict = {
        # origin
        'model': '/root/110/ultralytics-main/runs/train/exp15/weights/best.pt',
        'data':'/root/ultralytics-main/dataset/data-Copy1.yaml',
        'imgsz': 640,
        'epochs': 400,
        'batch': 16,
        'workers': 16,
        'cache': True,
        'optimizer': 'SGD',
        'device': '0',
        'close_mosaic': 20,
        'amp': False, # 如果蒸馏损失为nan，请把amp设置为False
        'project':'runs/distill',
        'name':'yolov8n-chsim-exp1',
        
        # distill
        'prune_model': False,
        'teacher_weights': '/root/110/ultralytics-main/runs/train/exp17/weights/best.pt',
        'teacher_cfg': '/root/110/ultralytics-main/ultralytics/cfg/models/v8/yolov8s-ASF-P2.yaml',
        'kd_loss_type': 'all',
        'kd_loss_decay': 'constant',
        
        'logical_loss_type': 'BCKD',
        'logical_loss_ratio': 1.0,
        
        'teacher_kd_layers': '31, 26, 21, 24',
        'student_kd_layers': '31, 26, 21, 24',
        'feature_loss_type': 'chsim',
        'feature_loss_ratio': 1.0
    }
    
    model = DetectionDistiller(overrides=param_dict)
    # model = SegmentationDistiller(overrides=param_dict)
    # model = PoseDistiller(overrides=param_dict)
    # model = OBBDistiller(overrides=param_dict)
    model.distill()