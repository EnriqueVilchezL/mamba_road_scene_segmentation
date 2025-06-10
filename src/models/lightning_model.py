import pytorch_lightning as pl
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

class SegmentationModel(pl.LightningModule):
    def __init__(
        self,
        model,
        loss_fn,
        metrics,
        lr,
        scheduler_max_it,
        weight_decay=0,
    ):
        super().__init__()
        self.model = model
        self.loss_fn = loss_fn

        self.train_metrics = metrics.clone(prefix="train/")
        self.val_metrics = metrics.clone(prefix="val/")
        self.test_metrics = metrics.clone(prefix="test/")

        self.scheduler_max_it = scheduler_max_it
        self.weight_decay = weight_decay
        self.lr = lr

    def forward(self, x):
        return self.model(x)

    def _common_step(self, batch, batch_idx):
        # x: [B, C, H, W], y: [B, H, W]
        x, y = batch
        # output: [B, num_classes, H, W]
        y_hat = self(x)

        loss = self.loss_fn(y_hat, y)
        return y, y_hat, loss
    
    def training_step(self, batch, batch_idx):
        # Forward pass
        y, y_hat, loss = self._common_step(batch, batch_idx)

        self.train_metrics.update(y_hat, y)

        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return {"loss": loss}
    
    def on_train_epoch_end(self):
        """
        Callback function called at the end of each training epoch.
        Computes and logs the training metrics.
        """
        self.log_dict(self.train_metrics.compute(), on_step=False, on_epoch=True)

        self.train_metrics.reset()

    def validation_step(self, batch, batch_idx):
        # Forward pass
        y, y_hat, loss = self._common_step(batch, batch_idx)

        self.val_metrics.update(y_hat, y)

        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return {"loss": loss}
    
    def on_validation_epoch_end(self):
        """
        Callback function called at the end of each validation epoch.
        Computes and logs the validation metrics.
        """
        self.log_dict(
            self.val_metrics.compute(), on_step=False, on_epoch=True, prog_bar=True
        )

        self.val_metrics.reset()
    
    def test_step(self, batch, batch_idx):
        # Forward pass
        y, y_hat, loss = self._common_step(batch, batch_idx)

        self.test_metrics.update(y_hat, y)

        self.log("test/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return {"loss": loss}
    
    def on_test_epoch_end(self):
        """
        Callback function called at the end of each testing epoch.
        Computes and logs the testing metrics.
        """
        self.log_dict(self.test_metrics.compute(), on_step=False, on_epoch=True)
        self.test_metrics.reset()

    def configure_optimizers(self):
        """
        Configure the optimizer and learning rate scheduler.

        Returns:
            Tuple containing the optimizer and learning rate scheduler.
        """
        optimizer = Adam(self.model.parameters(), lr=self.lr)
        scheduler = CosineAnnealingLR(optimizer, T_max=self.scheduler_max_it)
        return [optimizer], [scheduler]