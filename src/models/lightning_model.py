import pytorch_lightning as pl
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from explainability import get_segmentation_saliency, compute_feature_ablation
import wandb
import numpy as np
import torch
import matplotlib.pyplot as plt
from data_loader import class_to_rgb

class SegmentationModel(pl.LightningModule):
    def __init__(
        self,
        model,
        loss_fn,
        metrics,
        lr,
        class_names,
        scheduler_max_it,
        weight_decay=0,
        vectorized_metrics=None,
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

        self.class_names = class_names

        self.test_vectorized_metrics = None
        self.val_vectorized_metrics = None
        if vectorized_metrics is not None:
            self.test_vectorized_metrics = vectorized_metrics.clone(prefix="test/")
            self.val_vectorized_metrics = vectorized_metrics.clone(prefix="val/")

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

        if self.val_vectorized_metrics is not None:
            self.val_vectorized_metrics.update(y_hat, y)
        
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return {"loss": loss}
    
    def apply_colormap(self, saliency_map, cmap="inferno"):
        """
        Converts a 2D saliency map to a color heatmap using matplotlib colormap.
        """
        saliency_map = (saliency_map - saliency_map.min()) / (saliency_map.max() - saliency_map.min() + 1e-8)
        heatmap = plt.get_cmap(cmap)(saliency_map)[:, :, :3]  # Remove alpha channel
        heatmap = (heatmap * 255).astype(np.uint8)
        return heatmap


    def log_saliency_maps(self, model, mask, input_image, saliency_fn, class_names=None, cmap="inferno", logger=None, step=None):
        """
        Logs input image and saliency maps per class to Weights & Biases.
        
        Args:
            model (torch.nn.Module): The model used for prediction.
            input_image (Tensor): A single image of shape [1, C, H, W].
            saliency_fn (Callable): Function that computes saliency for a given class.
            class_names (List[str] or None): Optional class labels for captions.
            cmap (str): Matplotlib colormap to use for saliency.
            logger: wandb logger (`self.logger` in Lightning).
            step (int or None): Global step or epoch number to associate with the log.
        """
        if logger is None:
            return

        with torch.no_grad():
            output = model(input_image)
        num_classes = output.shape[1]

        input_np = input_image[0].permute(1, 2, 0).detach().cpu().numpy()
        input_np = (input_np - input_np.min()) / (input_np.max() - input_np.min() + 1e-8)

        saliency_images = []
        

        for cls in range(num_classes):
            binary_mask = (output.argmax(dim=1) == cls)

            sal_map = saliency_fn(model, input_image, mask=binary_mask, target_class=cls)  # expected: [H, W] numpy

            # Overlay on original image
            overlay = input_np.copy()
            heatmap = plt.get_cmap(cmap)(sal_map)[:, :, :3]  # RGB heatmap (no alpha)
            overlay = 0.3 * overlay + 0.7 * heatmap  # Blend original and saliency

            overlay = np.clip(overlay, 0, 1)
            label = class_names[cls] if class_names else f"Class {cls}"
            saliency_images.append(wandb.Image(overlay, caption=f"Saliency map overlayed: {label}"))

        print(mask.shape)
        logger.experiment.log({
            "Input Image": [wandb.Image(input_np, caption="Input Image"), wandb.Image(class_to_rgb(mask.permute(1, 2, 0).squeeze(-1).detach().cpu().numpy()), caption="Mask")],
            "Output Image": wandb.Image(class_to_rgb(output.argmax(dim=1)[0].detach().cpu().numpy()), caption="Output Image"),
            "Saliency Maps": saliency_images,
            "epoch": step
        })

    def log_feature_ablation_maps(
        self,
        model,
        input_image,
        feature_ablation_fn,
        mask,
        class_names=None,
        cmap="inferno",
        logger=None,
        step=None
    ):
        """
        Logs input image and feature ablation maps per class to Weights & Biases.

        Args:
            model (torch.nn.Module): The model used for prediction.
            input_image (Tensor): A single image of shape [1, C, H, W].
            feature_ablation_fn (Callable): Function to compute ablation per class.
                                            Must accept (model, input_image, class_index, mask) and return [H, W] numpy array.
            mask (Tensor): Mask tensor (e.g., output.argmax(1)), shape [1, H, W].
            class_names (List[str] or None): Optional class labels for captions.
            cmap (str): Matplotlib colormap.
            logger: W&B logger (e.g., self.logger).
            step (int or None): Global step or epoch for W&B logs.
        """
        if logger is None:
            return

        model.eval()

        # Prepare normalized input image for logging
        input_np = input_image[0].permute(1, 2, 0).detach().cpu().numpy()
        input_np = (input_np - input_np.min()) / (input_np.max() - input_np.min() + 1e-8)

        num_classes = model(input_image).shape[1]
        ablation_images = []

        for cls in range(num_classes):
            sal_map = feature_ablation_fn(model, input_image, cls, mask)  # <- renamed arg

            heatmap = self.apply_colormap(sal_map, cmap=cmap)
            label = class_names[cls] if class_names else f"Class {cls}"

            ablation_images.append(wandb.Image(heatmap, caption=f"Ablation: {label}"))

        logger.experiment.log({
            "Input Image (Ablation)": wandb.Image(input_np, caption="Input Image"),
            "Feature Ablation Maps": ablation_images,
            "epoch": step
        })

    def on_validation_epoch_end(self):
        self.log_dict(
            self.val_metrics.compute(), on_step=False, on_epoch=True, prog_bar=True
        )
        self.val_metrics.reset()

        if self.trainer.datamodule is not None:
            val_loader = self.trainer.datamodule.val_dataloader()
            sample_batch = next(iter(val_loader))
            x, y = sample_batch

            input_image = x[15:16].to(self.device)
            mask = y[15:16].to(self.device)
            # Create mask from ground truth y (one-hot or probs)
            # mask = y[0].argmax(dim=0, keepdim=True).to(self.device)  # [1, H, W]

            self.log_saliency_maps(
                model=self.model,
                mask=mask,
                input_image=input_image,
                saliency_fn=get_segmentation_saliency,
                class_names=self.class_names,
                cmap="inferno",
                logger=self.logger,
                step=self.current_epoch,
            )

            # Log feature ablation maps (new)
            # self.log_feature_ablation_maps(
            #     model=self.model,
            #     input_image=input_image,
            #     feature_ablation_fn=compute_feature_ablation,  # your ablation fn from utils
            #     mask=mask,
            #     class_names=self.class_names,
            #     cmap="inferno",
            #     logger=self.logger,
            #     step=self.current_epoch,
            # )

        if self.val_vectorized_metrics is not None:
            results = self.val_vectorized_metrics.compute()
            for metric_name, metric_tensor in results.items():
                
                if metric_tensor.ndim == 0:
                    # Scalar metric
                    wandb.log({metric_name: metric_tensor.item(), "epoch": self.current_epoch})
                else:
                    # Build wandb.Table
                    table = wandb.Table(columns=["Class", metric_name])
                    for i, val in enumerate(metric_tensor):
                        table.add_data(self.class_names[i], val.item())

                    wandb.log({
                        f"{metric_name}": wandb.plot.bar(
                            table, "Class", metric_name, title=metric_name
                        ),
                        "epoch": self.current_epoch
                    })

            self.test_vectorized_metrics.reset()
    
    def test_step(self, batch, batch_idx):
        # Forward pass
        y, y_hat, loss = self._common_step(batch, batch_idx)

        self.test_metrics.update(y_hat, y)

        if self.test_vectorized_metrics is not None:
            self.test_vectorized_metrics.update(y_hat, y)

        self.log("test/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return {"loss": loss}
    
    def on_test_epoch_end(self):
        """
        Callback function called at the end of each testing epoch.
        Computes and logs the testing metrics.
        """
        self.log_dict(self.test_metrics.compute(), on_step=False, on_epoch=True)
        self.test_metrics.reset()

        if self.test_vectorized_metrics is not None:
            results = self.test_vectorized_metrics.compute()  # Dict: metric_name → Tensor
            for metric_name, metric_tensor in results.items():
                
                # metric_tensor might be scalar (e.g. macro accuracy) or vector (per-class)
                if metric_tensor.ndim == 0:
                    # Scalar metric
                    wandb.log({metric_name: metric_tensor.item(), "epoch": self.current_epoch})
                else:
                    # Build wandb.Table
                    table = wandb.Table(columns=["Class", metric_name])
                    for i, val in enumerate(metric_tensor):
                        table.add_data(f"Class {i}", val.item())

                    wandb.log({
                        f"{metric_name}": wandb.plot.bar(
                            table, "Class", metric_name, title=metric_name
                        ),
                        "epoch": self.current_epoch
                    })

            self.test_vectorized_metrics.reset()

    def configure_optimizers(self):
        """
        Configure the optimizer and learning rate scheduler.

        Returns:
            Tuple containing the optimizer and learning rate scheduler.
        """
        optimizer = AdamW(self.model.parameters(), lr=self.lr, weight_decay=1e-3)
        scheduler = CosineAnnealingLR(optimizer, T_max=self.scheduler_max_it)
        return [optimizer], [scheduler]

class EnsembleSegmentationModel(pl.LightningModule):
    def __init__(
        self,
        models,
        loss_fn,
        metrics,
        lr,
        class_names,
        scheduler_max_it,
        weight_decay=0,
        vectorized_metrics=None,
    ):
        super().__init__()
        self.models = models
        self.loss_fn = loss_fn

        self.train_metrics = metrics.clone(prefix="train/")
        self.val_metrics = metrics.clone(prefix="val/")
        self.test_metrics = metrics.clone(prefix="test/")

        self.scheduler_max_it = scheduler_max_it
        self.weight_decay = weight_decay
        self.lr = lr

        self.class_names = class_names

        self.test_vectorized_metrics = None
        self.val_vectorized_metrics = None
        if vectorized_metrics is not None:
            self.test_vectorized_metrics = vectorized_metrics.clone(prefix="test/")
            self.val_vectorized_metrics = vectorized_metrics.clone(prefix="val/")

    def forward(self, x):
        # Collect soft predictions (after softmax) from all models
        preds = []
        for model in self.models:
            logits = model(x)  # shape: [B, C, H, W]
            preds.append(logits)

        # Stack and average: shape -> [num_models, B, C, H, W] → [B, C, H, W]
        avg_pred = torch.stack(preds, dim=0).mean(dim=0)
        return avg_pred

    def _common_step(self, batch, batch_idx):
        # x: [B, C, H, W], y: [B, H, W]
        x, y = batch
        # output: [B, num_classes, H, W]
        y_hat = self(x)

        loss = self.loss_fn(y_hat, y, epoch=self.current_epoch)
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

        if self.val_vectorized_metrics is not None:
            self.val_vectorized_metrics.update(y_hat, y)
        
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return {"loss": loss}
    
    def apply_colormap(self, saliency_map, cmap="inferno"):
        """
        Converts a 2D saliency map to a color heatmap using matplotlib colormap.
        """
        saliency_map = (saliency_map - saliency_map.min()) / (saliency_map.max() - saliency_map.min() + 1e-8)
        heatmap = plt.get_cmap(cmap)(saliency_map)[:, :, :3]  # Remove alpha channel
        heatmap = (heatmap * 255).astype(np.uint8)
        return heatmap


    def log_saliency_maps(self, model, mask, input_image, saliency_fn, class_names=None, cmap="inferno", logger=None, step=None):
        """
        Logs input image and saliency maps per class to Weights & Biases.
        
        Args:
            model (torch.nn.Module): The model used for prediction.
            input_image (Tensor): A single image of shape [1, C, H, W].
            saliency_fn (Callable): Function that computes saliency for a given class.
            class_names (List[str] or None): Optional class labels for captions.
            cmap (str): Matplotlib colormap to use for saliency.
            logger: wandb logger (`self.logger` in Lightning).
            step (int or None): Global step or epoch number to associate with the log.
        """
        if logger is None:
            return

        with torch.no_grad():
            output = model(input_image)
        num_classes = output.shape[1]

        input_np = input_image[0].permute(1, 2, 0).detach().cpu().numpy()
        input_np = (input_np - input_np.min()) / (input_np.max() - input_np.min() + 1e-8)

        saliency_images = []
        

        for cls in range(num_classes):
            binary_mask = (output.argmax(dim=1) == cls)

            sal_map = saliency_fn(model, input_image, mask=binary_mask, target_class=cls)  # expected: [H, W] numpy

            # Overlay on original image
            overlay = input_np.copy()
            heatmap = plt.get_cmap(cmap)(sal_map)[:, :, :3]  # RGB heatmap (no alpha)
            overlay = 0.3 * overlay + 0.7 * heatmap  # Blend original and saliency

            overlay = np.clip(overlay, 0, 1)
            label = class_names[cls] if class_names else f"Class {cls}"
            saliency_images.append(wandb.Image(overlay, caption=f"Saliency map overlayed: {label}"))

        print(mask.shape)
        logger.experiment.log({
            "Input Image": [wandb.Image(input_np, caption="Input Image"), wandb.Image(class_to_rgb(mask.permute(1, 2, 0).squeeze(-1).detach().cpu().numpy()), caption="Mask")],
            "Output Image": wandb.Image(class_to_rgb(output.argmax(dim=1)[0].detach().cpu().numpy()), caption="Output Image"),
            "Saliency Maps": saliency_images,
            "epoch": step
        })

    def log_feature_ablation_maps(
        self,
        model,
        input_image,
        feature_ablation_fn,
        mask,
        class_names=None,
        cmap="inferno",
        logger=None,
        step=None
    ):
        """
        Logs input image and feature ablation maps per class to Weights & Biases.

        Args:
            model (torch.nn.Module): The model used for prediction.
            input_image (Tensor): A single image of shape [1, C, H, W].
            feature_ablation_fn (Callable): Function to compute ablation per class.
                                            Must accept (model, input_image, class_index, mask) and return [H, W] numpy array.
            mask (Tensor): Mask tensor (e.g., output.argmax(1)), shape [1, H, W].
            class_names (List[str] or None): Optional class labels for captions.
            cmap (str): Matplotlib colormap.
            logger: W&B logger (e.g., self.logger).
            step (int or None): Global step or epoch for W&B logs.
        """
        if logger is None:
            return

        model.eval()

        # Prepare normalized input image for logging
        input_np = input_image[0].permute(1, 2, 0).detach().cpu().numpy()
        input_np = (input_np - input_np.min()) / (input_np.max() - input_np.min() + 1e-8)

        num_classes = model(input_image).shape[1]
        ablation_images = []

        for cls in range(num_classes):
            sal_map = feature_ablation_fn(model, input_image, cls, mask)  # <- renamed arg

            heatmap = self.apply_colormap(sal_map, cmap=cmap)
            label = class_names[cls] if class_names else f"Class {cls}"

            ablation_images.append(wandb.Image(heatmap, caption=f"Ablation: {label}"))

        logger.experiment.log({
            "Input Image (Ablation)": wandb.Image(input_np, caption="Input Image"),
            "Feature Ablation Maps": ablation_images,
            "epoch": step
        })

    def on_validation_epoch_end(self):
        self.log_dict(
            self.val_metrics.compute(), on_step=False, on_epoch=True, prog_bar=True
        )
        self.val_metrics.reset()

        if self.trainer.datamodule is not None:
            val_loader = self.trainer.datamodule.val_dataloader()
            sample_batch = next(iter(val_loader))
            x, y = sample_batch

            input_image = x[15:16].to(self.device)
            mask = y[15:16].to(self.device)
            # Create mask from ground truth y (one-hot or probs)
            # mask = y[0].argmax(dim=0, keepdim=True).to(self.device)  # [1, H, W]

            self.log_saliency_maps(
                model=self.model,
                mask=mask,
                input_image=input_image,
                saliency_fn=get_segmentation_saliency,
                class_names=self.class_names,
                cmap="inferno",
                logger=self.logger,
                step=self.current_epoch,
            )

            # Log feature ablation maps (new)
            # self.log_feature_ablation_maps(
            #     model=self.model,
            #     input_image=input_image,
            #     feature_ablation_fn=compute_feature_ablation,  # your ablation fn from utils
            #     mask=mask,
            #     class_names=self.class_names,
            #     cmap="inferno",
            #     logger=self.logger,
            #     step=self.current_epoch,
            # )

        if self.val_vectorized_metrics is not None:
            results = self.val_vectorized_metrics.compute()
            for metric_name, metric_tensor in results.items():
                
                if metric_tensor.ndim == 0:
                    # Scalar metric
                    wandb.log({metric_name: metric_tensor.item(), "epoch": self.current_epoch})
                else:
                    # Build wandb.Table
                    table = wandb.Table(columns=["Class", metric_name])
                    for i, val in enumerate(metric_tensor):
                        table.add_data(self.class_names[i], val.item())

                    wandb.log({
                        f"{metric_name}": wandb.plot.bar(
                            table, "Class", metric_name, title=metric_name
                        ),
                        "epoch": self.current_epoch
                    })

            self.test_vectorized_metrics.reset()
    
    def test_step(self, batch, batch_idx):
        # Forward pass
        y, y_hat, loss = self._common_step(batch, batch_idx)

        self.test_metrics.update(y_hat, y)

        if self.test_vectorized_metrics is not None:
            self.test_vectorized_metrics.update(y_hat, y)

        self.log("test/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return {"loss": loss}
    
    def on_test_epoch_end(self):
        """
        Callback function called at the end of each testing epoch.
        Computes and logs the testing metrics.
        """
        self.log_dict(self.test_metrics.compute(), on_step=False, on_epoch=True)
        self.test_metrics.reset()

        if self.test_vectorized_metrics is not None:
            results = self.test_vectorized_metrics.compute()  # Dict: metric_name → Tensor
            for metric_name, metric_tensor in results.items():
                
                # metric_tensor might be scalar (e.g. macro accuracy) or vector (per-class)
                if metric_tensor.ndim == 0:
                    # Scalar metric
                    wandb.log({metric_name: metric_tensor.item(), "epoch": self.current_epoch})
                else:
                    # Build wandb.Table
                    table = wandb.Table(columns=["Class", metric_name])
                    for i, val in enumerate(metric_tensor):
                        table.add_data(f"Class {i}", val.item())

                    wandb.log({
                        f"{metric_name}": wandb.plot.bar(
                            table, "Class", metric_name, title=metric_name
                        ),
                        "epoch": self.current_epoch
                    })

            self.test_vectorized_metrics.reset()

    def configure_optimizers(self):
        """
        Configure the optimizer and learning rate scheduler.

        Returns:
            Tuple containing the optimizer and learning rate scheduler.
        """
        optimizer = AdamW(self.model.parameters(), lr=self.lr, weight_decay=1e-3)
        scheduler = CosineAnnealingLR(optimizer, T_max=self.scheduler_max_it)
        return [optimizer], [scheduler]