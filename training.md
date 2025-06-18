# Model Training

This section describes the shared training setup used across all models (UNet, Swin-UNet, and Mamba-UNet). All models were trained under the exact same experimental conditions to ensure fair comparison and reproducibility.

And change the `WANDB_PROJECT = "mamba_road_scene_segmentation"` to your wandb project name.

Each model is trained via its respective script:

```bash
src/  
└── experiments/  
  ├── e0_unet.py        # Training script for UNet  
  ├── e1_swin_unet.py   # Training script for Swin-UNet  
  └── e2_mamba_unet.py  # Training script for Mamba-UNet  
```

---

## General Methodology

The following configuration was used for training all models:

| Setting                 | Value                     |
|-------------------------|---------------------------|
| Hardware                | Tesla V100S-PCIE-32GB     |
| Optimizer               | AdamW                     |
| Learning Rate           | 0.0003                    |
| Weight Decay            | 1e-3                      |
| Scheduler               | Cosine Annealing          |
| Batch Size              | 24                        |
| Max Epochs              | 100                       |
| Early Stopping Patience | 10                        |
| Input Image Size        | 224 × 224                 |
| Random Seed             | 42 (PyTorch & NumPy)      |

---

## Pipeline

The training procedure is structured as follows:

---

### 1. Initial Imports

This block sets up relative imports and loads all required libraries.

```python
import os  
import sys  
import inspect  

currentdir = os.path.dirname(  
  os.path.abspath(inspect.getfile(inspect.currentframe()))  
)  
parentdir = os.path.dirname(currentdir)  
grandparentdir = os.path.dirname(parentdir)  

for path in [parentdir, grandparentdir]:  
  if path not in sys.path:  
    sys.path.insert(0, path)  

# Core imports  
import torch.nn as nn  
import torch  
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping  
from pytorch_lightning.loggers import WandbLogger  
from pytorch_lightning import Trainer  
from torchmetrics import MetricCollection  
from torchmetrics.classification import MulticlassAccuracy, MulticlassJaccardIndex  

# Custom modules  
from data_loader import SegmentationDataModule, SegmentationTrainTransform, SegmentationTestTransform  
from models.unet import UNet  
from models.lightning_model import SegmentationModel  
from utils import get_device, set_seed  
from loss import CombinedLoss, SymmetricUnifiedFocalLoss  
import configuration as config  
import wandb  
```

---

### 2. Utility Setup

These utility methods configure the device and random seed.

```python
device = get_device()  
set_seed()  
```

Setting a seed assures reproducibility, since it initializes the weights the same way each time.
---

### 3. Data Module Setup

This sets up the dataloader with the specified transformations.

```python
datamodule = SegmentationDataModule(  
  batch_size=config.BATCH_SIZE,  
  data_path=config.DATA_PATH,  
  num_classes=config.NUM_CLASSES,  
  train_transform=SegmentationTrainTransform(  
    image_size=config.UNET_IMAGE_SIZE  
  ),  
  test_transform=SegmentationTestTransform(  
    image_size=config.UNET_IMAGE_SIZE  
  ),  
  num_workers=config.NUM_WORKERS,  
)
```

Also, the `setup` method configurates the dataloader for training and validation.

```python
datamodule.setup()  
```

### 4. Transformations

We did data augmentation, and made sure that the image and mask pairs get the same transformations when needed. That is why they share some such as the random rotations, crops or flips.

**Training Transforms**

| Type   | Transformations                                                                 |
|--------|----------------------------------------------------------------------------------|
| Images | Resize (256×256), RandomHorizontalFlip (p=0.8), RandomCrop, RandomRotation (30°), ColorJitter, GaussianBlur, AddGaussianNoise, ToTensor, Normalize ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) |
| Masks  | Resize (224×224), RandomHorizontalFlip (p=0.8), RandomCrop, RandomRotation (30°), ToTensor |

**Testing Transforms**

| Type   | Transformations                                       |
|--------|--------------------------------------------------------|
| Images | Resize (256×256), ToTensor, Normalize (same as above) |
| Masks  | Resize (224×224, NEAREST interpolation)               |

---

### 4. Defining Metrics

We used the following metrics to evaluate model performance:

| Metric             | Description                                                                 |
|--------------------|-----------------------------------------------------------------------------|
| Accuracy           | Micro-averaged classification accuracy across all pixels and classes        |
| Balanced Accuracy  | Macro-averaged accuracy; treats all classes equally regardless of frequency |
| IoU                | Mean Intersection over Union (IoU) per class, averaged (macro)              |
| PerClassAccuracy   | Accuracy for each individual class (no averaging)                           |
| PerClassIoU        | IoU score for each class (no averaging)                                     |

```python
metrics = MetricCollection({  
  "Accuracy": MulticlassAccuracy(num_classes=config.NUM_CLASSES, average="micro"),  
  "BalancedAccuracy": MulticlassAccuracy(num_classes=config.NUM_CLASSES, average="macro"),  
  "IoU": MulticlassJaccardIndex(num_classes=config.NUM_CLASSES, average="macro"),  
})  

vectorized_metrics = MetricCollection({  
  "PerClassAccuracy": MulticlassAccuracy(num_classes=config.NUM_CLASSES, average="none"),  
  "PerClassIoU": MulticlassJaccardIndex(num_classes=config.NUM_CLASSES, average="none"),  
})  
```

---

### 5. Model Setup

The model is instantiated and wrapped in a PyTorch Lightning module for training.

```python
unet = UNet(in_channels=3, num_classes=config.NUM_CLASSES)  

model = SegmentationModel(  
  model=unet,  
  lr=config.LR,  
  class_names=list(config.LABEL_MAP.values()),  
  metrics=metrics,  
  vectorized_metrics=vectorized_metrics,  
  loss_fn=SymmetricUnifiedFocalLoss(),  
  scheduler_max_it=config.SCHEDULER_MAX_IT,  
)
```

---

### 6. Callbacks and Logging

We configure early stopping, model checkpointing.

```python
early_stop_callback = EarlyStopping(  
  monitor="val/loss",  
  patience=config.PATIENCE,  
  strict=False,  
  verbose=False,  
  mode="min",  
)  

checkpoint_callback = ModelCheckpoint(  
  monitor="val/loss",  
  dirpath=config.UNET_CHECKPOINT_PATH,  
  filename=config.UNET_FILENAME,  
  save_top_k=config.TOP_K_SAVES,  
  mode="min",  
)  
```

We also configured wandb logging.

```python
id = "unet_ufl_1000"  
wandb_logger = WandbLogger(project=config.WANDB_PROJECT, id=id, resume="allow")  
```

---

### 7. Training and Testing

```python
trainer = Trainer(  
  logger=wandb_logger,  
  max_epochs=config.EPOCHS,  
  accelerator=device,  
  callbacks=[checkpoint_callback, early_stop_callback],  
  log_every_n_steps=1,  
)  

trainer.fit(model, datamodule=datamodule)  
```

After that, we load the best weights saved by the `ModelCheckpoint` callback to test it and get metrics.

```python
# Reload best checkpoint for testing  
unet = UNet(in_channels=3, num_classes=config.NUM_CLASSES)  

model = SegmentationModel.load_from_checkpoint(  
  config.UNET_CHECKPOINT_PATH + "/" + config.UNET_FILENAME,  
  model=unet,  
  lr=config.LR,  
  class_names=list(config.LABEL_MAP.values()),  
  metrics=metrics,  
  vectorized_metrics=vectorized_metrics,  
  loss_fn=SymmetricUnifiedFocalLoss(),  
  scheduler_max_it=config.SCHEDULER_MAX_IT,  
)  

trainer.test(model, datamodule=datamodule)  
```

---

### 8. Entry Point

```
if __name__ == "__main__":  
  main()  
```

---

This setup ensures that all training runs are fully reproducible, consistently measured, and tracked using modern best practices.