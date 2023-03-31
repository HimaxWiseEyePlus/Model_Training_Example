from easydict import EasyDict as edict


__C = edict()
# Consumers can get config by: from config import cfg
cfg = __C


# YOLO options
__C.YOLO = edict()
__C.YOLO.darknettxt = False
__C.YOLO.ignore_thresh = 0.5
__C.YOLO.iou_thresh = 0.213
__C.YOLO.L2_FACTOR = 0.0005
__C.YOLO.ACT_TYPE = 'relu6' # leaky relu relu6
__C.YOLO.max_boxes = 200
__C.YOLO.warmup_epochs = 1
__C.YOLO.input_shape = (320, 320, 3) # (None, None, 3)
__C.YOLO.fixshape = False
__C.YOLO.grayscale = True

# MOT options
__C.MOT = edict()
# pruning
__C.MOT.final_sparsity = 0.5
__C.MOT.frequency = 3696
# cluster
__C.MOT.number_of_clusters = 16
__C.MOT.preserve_sparsity = True
__C.MOT.cluster_centroids_init = 'KMEANS_PLUS_PLUS' # LINEAR # KMEANS_PLUS_PLUS # DENSITY_BASED
#QAT
__C.MOT.qat_mode = 'pcqat' # qat pqat pcqat cqat