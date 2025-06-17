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
    import torch
    from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
    from pytorch_lightning.loggers import WandbLogger
    from pytorch_lightning import Trainer
    from torchmetrics import MetricCollection
    from torchmetrics.classification import MulticlassAccuracy, MulticlassJaccardIndex

    from data_loader import SegmentationDataModule, SegmentationTrainTransform, SegmentationTestTransform
    from models.mamba_unet import MambaUnet
    from models.swin_unet import SwinUnet
    from models.unet import UNet
    from models.lightning_model import SegmentationModel, EnsembleSegmentationModel

    from utils import get_device
    from loss import CombinedLoss, SymmetricUnifiedFocalLoss
    import configuration as config
    import wandb

    # Get the available device
    device = get_device()

    # Get the training and validation datasets
    datamodule = SegmentationDataModule(
        batch_size=config.BATCH_SIZE,
        data_path=config.DATA_PATH,
        num_classes=config.NUM_CLASSES,
        train_transform=SegmentationTrainTransform(
            image_size=config.MAMBA_IMAGE_SIZE
        ),
        test_transform=SegmentationTestTransform(
            image_size=config.MAMBA_IMAGE_SIZE
        ),
        num_workers=config.NUM_WORKERS,
    )
    datamodule.setup()

    metrics = MetricCollection(
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
    vectorized_metrics = MetricCollection(
        {
            "PerClassAccuracy": MulticlassAccuracy(
                num_classes=config.NUM_CLASSES, average="none"
            ),
            "PerClassIoU": MulticlassJaccardIndex(
                num_classes=config.NUM_CLASSES, average="none"
            ),
        }
    )

    mamba = MambaUnet(img_size=config.MAMBA_IMAGE_SIZE[0], num_classes=config.NUM_CLASSES)
    model_mamba = SegmentationModel.load_from_checkpoint(
        "results/mamba_unet/checkpoints/mamba_unet_model.pth.ckpt",
        model=mamba,
        lr=config.LR,
        class_names=list(config.LABEL_MAP.values()),
        metrics=metrics,
        vectorized_metrics=vectorized_metrics,
        loss_fn=CombinedLoss(epochs=config.EPOCHS),
        scheduler_max_it=config.SCHEDULER_MAX_IT,
    ).to(torch.device(device))

    swin = SwinUnet(img_size=config.SWIN_IMAGE_SIZE[0], num_classes=config.NUM_CLASSES)
    model_swin = SegmentationModel.load_from_checkpoint(
        "results/swin_unet/swin_unet_model.pth.ckpt",
        model=swin,
        lr=config.LR,
        class_names=list(config.LABEL_MAP.values()),
        metrics=metrics,
        vectorized_metrics=vectorized_metrics,
        loss_fn=CombinedLoss(epochs=config.EPOCHS),
        scheduler_max_it=config.SCHEDULER_MAX_IT,
    ).to(torch.device(device))

    unet = UNet(in_channels=3, num_classes=config.NUM_CLASSES)
    model_unet = SegmentationModel.load_from_checkpoint(
        "results/unet/checkpoints/unet_model.pth.ckpt",
        model=unet,
        lr=config.LR,
        class_names=list(config.LABEL_MAP.values()),
        metrics=metrics,
        vectorized_metrics=vectorized_metrics,
        loss_fn=CombinedLoss(epochs=config.EPOCHS),
        scheduler_max_it=config.SCHEDULER_MAX_IT,
    ).to(torch.device(device))

    ensemble = EnsembleSegmentationModel(
        models=[model_swin, model_mamba],
        lr=config.LR,
        class_names=list(config.LABEL_MAP.values()),
        metrics=metrics,
        vectorized_metrics=vectorized_metrics,
        loss_fn=CombinedLoss(epochs=config.EPOCHS),
        scheduler_max_it=config.SCHEDULER_MAX_IT,
    )

    id = "ensemble_swin_mamba"
    wandb_logger = WandbLogger(project=config.WANDB_PROJECT, id=id, resume="allow")
    trainer = Trainer(
        logger=wandb_logger,
        max_epochs=config.EPOCHS,
        accelerator=device,
        log_every_n_steps=1,
    )
    trainer.test(ensemble, datamodule=datamodule)


if __name__ == "__main__":
    main()

