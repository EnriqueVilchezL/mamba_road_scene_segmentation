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

    from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
    from pytorch_lightning.loggers import WandbLogger
    from pytorch_lightning import Trainer
    from torchmetrics import MetricCollection
    from torchmetrics.classification import MulticlassAccuracy, MulticlassJaccardIndex

    from data_loader import (
        SegmentationDataModule,
        SegmentationTrainTransform,
        SegmentationValTransform,
    )
    from models.hmt_unet import HMTUNet
    from models.lightning_model import SegmentationModel

    from utils import get_device, set_seed
    from loss import SymmetricUnifiedFocalLoss
    import configuration as config

    # Get the available device
    device = get_device()
    set_seed()

    # Get the training and validation datasets
    datamodule = SegmentationDataModule(
        batch_size=config.BATCH_SIZE,
        data_path=config.DATA_PATH,
        num_classes=config.NUM_CLASSES,
        train_transform=SegmentationTrainTransform(image_size=config.HMT_IMAGE_SIZE),
        val_transform=SegmentationValTransform(image_size=config.HMT_IMAGE_SIZE),
        test_transform=SegmentationValTransform(image_size=config.HMT_IMAGE_SIZE),
        num_workers=config.NUM_WORKERS,
    )
    datamodule.setup()

    metrics = MetricCollection(
        {
            "PA": MulticlassAccuracy(num_classes=config.NUM_CLASSES, average="micro"),
            "mPA": MulticlassAccuracy(num_classes=config.NUM_CLASSES, average="macro"),
            "mIoU": MulticlassJaccardIndex(
                num_classes=config.NUM_CLASSES, average="macro"
            ),
        }
    )
    vectorized_metrics = MetricCollection(
        {
            "PA": MulticlassAccuracy(num_classes=config.NUM_CLASSES, average="none"),
            "IoU": MulticlassJaccardIndex(
                num_classes=config.NUM_CLASSES, average="none"
            ),
        }
    )

    hybrid = HMTUNet(
        num_classes=config.NUM_CLASSES, resolution=config.HMT_IMAGE_SIZE[0]
    )
    model = SegmentationModel(
        model=hybrid,
        lr=config.LR,
        class_names=list(config.LABEL_MAP.values()),
        metrics=metrics,
        vectorized_metrics=vectorized_metrics,
        loss_fn=SymmetricUnifiedFocalLoss(),
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
        dirpath=config.HMT_UNET_CHECKPOINT_PATH,
        filename=config.HMT_UNET_FILE_NAME,
        save_top_k=config.TOP_K_SAVES,
        mode="min",
    )

    id = "hmt_unet_final"
    wandb_logger = WandbLogger(project=config.WANDB_PROJECT, id=id, resume="allow")
    trainer = Trainer(
        logger=wandb_logger,
        max_epochs=config.EPOCHS,
        accelerator=device,
        callbacks=[checkpoint_callback, early_stop_callback],
        log_every_n_steps=1,
    )
    trainer.fit(model, datamodule=datamodule)

    # Test
    hybrid = HMTUNet(
        num_classes=config.NUM_CLASSES, resolution=config.HMT_IMAGE_SIZE[0]
    )
    model = SegmentationModel.load_from_checkpoint(
        config.HMT_UNET_CHECKPOINT_PATH + "/" + config.HMT_UNET_FILE_NAME + ".ckpt",
        model=hybrid,
        lr=config.LR,
        class_names=list(config.LABEL_MAP.values()),
        metrics=metrics,
        vectorized_metrics=vectorized_metrics,
        loss_fn=SymmetricUnifiedFocalLoss(),
        scheduler_max_it=config.SCHEDULER_MAX_IT,
    )
    trainer.test(model, datamodule=datamodule)


if __name__ == "__main__":
    main()
