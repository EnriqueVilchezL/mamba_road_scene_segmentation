# Model Training

This section details the unified training pipeline used for all models (UNet, Swin-UNet, Mamba-UNet). By using the exact same setup across experiments, we ensure **fair comparison, consistent evaluation, and full reproducibility**.

Before you run any training scripts, update the `WANDB_PROJECT` variable in your config to match your **actual WandB project name**:

```python
WANDB_PROJECT = "your_actual_wandb_project_name"
```

Each model has a dedicated training script located in `src/experiments`:

```bash
src/  
└── experiments/  
    ├── e0_unet.py        # Training script for UNet  
    ├── e1_swin_unet.py   # Training script for Swin-UNet  
    └── e2_mamba_unet.py  # Training script for Mamba-UNet  
```

---

## General Methodology

The training configuration is summarized below. These values were fixed across all model variants.

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

This ensures models are trained under identical conditions for fair benchmarking.

---

## Pipeline Overview

The training and evaluation process is divided into the following stages:

---

### 1. Initial Setup

First, import all necessary modules and set up the environment. This includes adding parent directories to `sys.path` for relative imports, and configuring device and seed.

```python
device = get_device()  
set_seed()  
```

Using a fixed seed guarantees reproducibility in weight initialization, data shuffling, and augmentation.

---

### 2. Data Loading

We use a custom `SegmentationDataModule` to manage data loading and preprocessing for training, validation, and testing.

```python
datamodule = SegmentationDataModule(
    batch_size=config.BATCH_SIZE,
    data_path=config.DATA_PATH,
    num_classes=config.NUM_CLASSES,
    train_transform=SegmentationTrainTransform(image_size=config.UNET_IMAGE_SIZE),
    val_transform=SegmentationValTransform(image_size=config.UNET_IMAGE_SIZE),
    test_transform=SegmentationValTransform(image_size=config.UNET_IMAGE_SIZE),
    num_workers=config.NUM_WORKERS,
)
datamodule.setup()
```

This encapsulates all dataloader logic, including augmentations and batching.

---

### 3. Data Augmentation

Data augmentation is applied to improve generalization and robustness. All images and masks are transformed **consistently** to maintain pixel-wise alignment.

**Training Augmentations**

| Type   | Transformations                                                                 |
|--------|----------------------------------------------------------------------------------|
| Images | Resize (256×256), RandomHorizontalFlip (p=0.8), RandomCrop, RandomRotation (30°), ColorJitter, GaussianBlur, AddGaussianNoise, ToTensor, Normalize ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) |
| Masks  | Resize (224×224), RandomHorizontalFlip (p=0.8), RandomCrop, RandomRotation (30°), ToTensor |

**Validation & Testing**

| Type   | Transformations                                       |
|--------|--------------------------------------------------------|
| Images | Resize (256×256), ToTensor, Normalize (same as above) |
| Masks  | Resize (224×224, NEAREST interpolation)               |

---

### 4. Metrics Configuration

We define both aggregate (macro/micro) and per-class metrics using `torchmetrics`. These metrics are logged throughout training and evaluation.

```python
metrics = MetricCollection({
    "PA": MulticlassAccuracy(num_classes=config.NUM_CLASSES, average="micro"),
    "mPA": MulticlassAccuracy(num_classes=config.NUM_CLASSES, average="macro"),
    "mIoU": MulticlassJaccardIndex(num_classes=config.NUM_CLASSES, average="macro"),
})

vectorized_metrics = MetricCollection({
    "PA": MulticlassAccuracy(num_classes=config.NUM_CLASSES, average="none"),
    "IoU": MulticlassJaccardIndex(num_classes=config.NUM_CLASSES, average="none"),
})
```

These metrics help interpret model performance from both overall and class-specific perspectives.

---

### 5. Model Setup

Here we initialize the UNet model and wrap it in a LightningModule along with our loss function and metrics.

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

The loss function is a custom **Symmetric Unified Focal Loss**, designed to handle class imbalance effectively.

---

### 6. Callbacks and Logging

We set up early stopping and checkpoint saving:

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

Then configure logging with Weights & Biases:

```python
id = "unet_final_original"
wandb_logger = WandbLogger(project=config.WANDB_PROJECT, id=id, resume="allow")
```

---

### 7. Training the Model

We now launch training using PyTorch Lightning’s `Trainer`:

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

---

### 8. Testing the Model

Once training is complete, we reload the best saved model and run evaluation:

```python
unet = UNet(in_channels=3, num_classes=config.NUM_CLASSES)
model = SegmentationModel.load_from_checkpoint(
    config.UNET_CHECKPOINT_PATH + "/" + config.UNET_FILENAME + ".ckpt",
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

This generates the final evaluation metrics across the test set.

---

### 9. Script Entry Point

Wrap everything in a main function:

```python
if __name__ == "__main__":
    main()
```

---
