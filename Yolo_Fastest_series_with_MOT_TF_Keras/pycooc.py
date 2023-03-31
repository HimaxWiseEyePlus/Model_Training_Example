import os, argparse
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import numpy as np

parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS, description='evaluate 1 channel YOLO model (tflite) with test dataset')
'''
Command line options
'''
parser.add_argument(
    '--res_path', type=str, required=True,
    help='path to resoults file')

parser.add_argument(
    '--instances_json_file', type=str, required=True,
    help='COCO instances json path')

annType = ['segm','bbox','keypoints']
annType = annType[1]      #specify type here

#initialize COCO ground truth api
args = parser.parse_args()
annFile = args.instances_json_file
cocoGt=COCO(annFile)

#initialize COCO detections api
resFile= args.res_path
cocoDt=cocoGt.loadRes(resFile)

imgIds = cocoGt.getImgIds()


# running evaluation
cocoEval = COCOeval(cocoGt,cocoDt,annType)
cocoEval.params.imgIds  = imgIds
#cocoEval.params.catIds  = catIds
cocoEval.evaluate()
cocoEval.accumulate()
cocoEval.summarize()