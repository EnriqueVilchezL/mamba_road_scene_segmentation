NUM_CLASSES = 20
DATA_PATH = "data"
BATCH_SIZE = 24

PATIENCE = 10
LR = 0.0003
SCHEDULER_MAX_IT = 100
WEIGH_DECAY = 1e-3
EPSILON = 1e-4
EPOCHS = 100
TOP_K_SAVES = 1
NUM_WORKERS = 71

WANDB_PROJECT = "mamba_road_scene_segmentation"

UNET_IMAGE_SIZE = (224, 224)
UNET_RESULTS_PATH = "results/unet"
UNET_FILENAME = "unet_model.pth"
UNET_CHECKPOINT_PATH = f"{UNET_RESULTS_PATH}/checkpoints"

SWIN_IMAGE_SIZE = (448, 448)
SWIN_RESULTS_PATH = "results/swin_unet"
SWIN_UNET_FILENAME = "swin_unet_model.pth"
SWIN_UNET_CHECKPOINT_PATH = f"{SWIN_RESULTS_PATH}/checkpoints"

MAMBA_IMAGE_SIZE = (224, 224)
MAMBA_UNET_RESULTS_PATH = "results/mamba_unet"
MAMBA_UNET_FILENAME = "mamba_unet_model.pth"
MAMBA_UNET_CHECKPOINT_PATH = f"{MAMBA_UNET_RESULTS_PATH}/checkpoints"

HMT_IMAGE_SIZE = (224, 224)
HMT_UNET_RESULTS_PATH = "results/hmt_unet"
HMT_UNET_FILE_NAME = "hmt_unet_model.pth"
HMT_UNET_CHECKPOINT_PATH = f"{HMT_UNET_RESULTS_PATH}/checkpoints"

COLOR_MAP = {
    (128, 64, 128): 0,  # road
    (244, 35, 232): 1,  # sidewalk
    (70, 70, 70): 2,  # building
    (102, 102, 156): 3,  # wall
    (190, 153, 153): 4,  # fence
    (153, 153, 153): 5,  # pole
    (250, 170, 30): 6,  # traffic light
    (220, 220, 0): 7,  # traffic sign
    (107, 142, 35): 8,  # vegetation
    (152, 251, 152): 9,  # terrain
    (70, 130, 180): 10,  # sky
    (220, 20, 60): 11,  # person
    (255, 0, 0): 12,  # rider
    (0, 0, 142): 13,  # car
    (0, 0, 70): 14,  # truck
    (0, 60, 100): 15,  # bus
    (0, 80, 100): 16,  # train
    (0, 0, 230): 17,  # motorcycle
    (119, 11, 32): 18,  # bicycle
    (0, 0, 0): 19,  # unknown / background
}

LABEL_MAP = {
    0: "road",
    1: "sidewalk",
    2: "building",
    3: "wall",
    4: "fence",
    5: "pole",
    6: "traffic light",
    7: "traffic sign",
    8: "vegetation",
    9: "terrain",
    10: "sky",
    11: "person",
    12: "rider",
    13: "car",
    14: "truck",
    15: "bus",
    16: "train",
    17: "motorcycle",
    18: "bicycle",
    19: "unknown",
}
