# sin filtro
def main():
    import os
    import sys
    import inspect

    currentdir = os.path.dirname(
        os.path.abspath(inspect.getfile(inspect.currentframe()))
    )
    parentdir = os.path.dirname(currentdir)
    grandparentdir = os.path.dirname(parentdir)

    # Insert them into sys.path
    for path in [parentdir, grandparentdir]:
        if path not in sys.path:
            sys.path.insert(0, path)

    import torch.nn as nn
    from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
    from pytorch_lightning.loggers import WandbLogger
    from pytorch_lightning import Trainer
    from torchmetrics import MetricCollection
    from torchmetrics.classification import MulticlassAccuracy, MulticlassJaccardIndex

    from data_loader import SegmentationDataModule, SegmentationTransform
    from models.mamba_unet import MambaUnet
    from models.lightning_model import SegmentationModel

    from utils import get_device
    import configuration as config
    import wandb

    # Get the available device
    device = get_device()

    # Get the training and validation datasets
    datamodule = SegmentationDataModule(
        batch_size=config.BATCH_SIZE,
        data_path=config.DATA_PATH,
        num_classes=config.NUM_CLASSES,
        transform=SegmentationTransform(
            image_size=config.MAMBA_IMAGE_SIZE
        ),
        num_workers=config.NUM_WORKERS,
    )
    datamodule.setup()

    metrics = MetricCollection(
        {
            "Accuracy": MulticlassAccuracy(
                num_classes=config.NUM_CLASSES, average="macro"
            ),
            "IoU": MulticlassJaccardIndex(
                num_classes=config.NUM_CLASSES, average="macro"
            ),
        }
    )
    vectorized_metrics = MetricCollection(
        {
            "Accuracy": MulticlassAccuracy(
                num_classes=config.NUM_CLASSES, average="micro"
            ),
            "BalancedAccuracy": MulticlassAccuracy(
                num_classes=config.NUM_CLASSES, average="macro"
            ),
            "IoU": MulticlassJaccardIndex(
                num_classes=config.NUM_CLASSES, average="macro"
            ),
        }
    )

    mamba = MambaUnet(img_size=config.MAMBA_IMAGE_SIZE[0], num_classes=config.NUM_CLASSES)
    model = SegmentationModel(
        model=mamba,
        lr=config.LR,
        class_names=list(config.LABEL_MAP.values()),
        metrics=metrics,
        vectorized_metrics=vectorized_metrics,
        loss_fn=nn.CrossEntropyLoss(),
        scheduler_max_it=config.SCHEDULER_MAX_IT,
    )

    early_stop_callback = EarlyStopping(
        monitor="val/loss",
        patience=config.PATIENCE,
        strict=False,
        verbose=False,
        mode="min",
    )

    checkpoint_callback = ModelCheckpoint(
        monitor="val/loss",
        dirpath=config.MAMBA_UNET_CHECKPOINT_PATH,
        filename=config.MAMBA_UNET_FILENAME,
        save_top_k=config.TOP_K_SAVES,
        mode="min",
    )

    id = config.MAMBA_UNET_FILENAME.split(".")[0] + "_" + wandb.util.generate_id()
    wandb_logger = WandbLogger(project=config.WANDB_PROJECT, id=id, resume="allow")
    trainer = Trainer(
        logger=wandb_logger,
        max_epochs=config.EPOCHS,
        accelerator=device,
        callbacks=[checkpoint_callback, early_stop_callback],
        log_every_n_steps=1,
    )
    trainer.fit(model, datamodule=datamodule)


if __name__ == "__main__":
    main()

