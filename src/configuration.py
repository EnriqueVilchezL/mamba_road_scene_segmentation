NUM_CLASSES = 12
DATA_PATH = "data"
BATCH_SIZE = 32

PATIENCE = 3
LR = 0.0003
SCHEDULER_MAX_IT = 30
WEIGH_DECAY = 1e-4
EPSILON = 1e-4
EPOCHS = 15
TOP_K_SAVES = 1
NUM_WORKERS = 1

IMAGE_SIZE = (256, 256)
WANDB_PROJECT = "mamba_road_scene_segmentation"

UNET_RESULTS_PATH = "results/unet"
UNET_FILENAME = "unet_model.pth"
UNET_CHECKPOINT_PATH = f"{UNET_RESULTS_PATH}/checkpoints"

COLOR_MAP = {
    (70, 130, 180): 0,  # sky
    (70, 70, 70): 1,  # building
    (153, 153, 153): 2,  # pole
    (128, 64, 128): 3,  # road
    (244, 35, 232): 4,  # pavement
    (107, 142, 35): 5,  # tree
    (220, 220, 0): 6,  # signsymbol
    (190, 153, 153): 7,  # fence
    (0, 0, 142): 8,  # car
    (220, 20, 60): 9,  # pedestrian
    (0, 0, 230): 10,  # bicyclist
    (0, 0, 0): 11,  # unlabelled / background
}

LABEL_MAP = {
    0: "sky",
    1: "building",
    2: "pole",
    3: "road",
    4: "pavement",
    5: "tree",
    6: "signsymbol",
    7: "fence",
    8: "car",
    9: "pedestrian",
    10: "bicyclist",
    11: "unlabelled / background",
}