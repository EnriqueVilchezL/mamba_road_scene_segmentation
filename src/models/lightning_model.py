import pytorch_lightning as pl
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from explainability import get_segmentation_saliency
import wandb
import numpy as np
import torch
import matplotlib.pyplot as plt
from data_loader import class_to_rgb
from time import perf_counter


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
        """
        Initializes the Lightning model wrapper.

        Args:
            model (torch.nn.Module): The neural network model to be trained and evaluated.
            loss_fn (callable): The loss function used for optimization.
            metrics (torchmetrics.MetricCollection): Metric collection for training, validation, and testing.
            lr (float): Learning rate for the optimizer.
            class_names (list): List of class names for segmentation or classification tasks.
            scheduler_max_it (int): Maximum number of iterations for the learning rate scheduler.
            weight_decay (float, optional): Weight decay (L2 regularization) factor. Defaults to 0.
            vectorized_metrics (torchmetrics.MetricCollection, optional): Additional metrics that operate in a vectorized manner for validation and testing. Defaults to None.

        Attributes:
            model (torch.nn.Module): The neural network model.
            loss_fn (callable): The loss function.
            train_metrics (torchmetrics.MetricCollection): Metrics for training phase.
            val_metrics (torchmetrics.MetricCollection): Metrics for validation phase.
            test_metrics (torchmetrics.MetricCollection): Metrics for testing phase.
            scheduler_max_it (int): Maximum iterations for scheduler.
            weight_decay (float): Weight decay factor.
            lr (float): Learning rate.
            class_names (list): List of class names.
            test_vectorized_metrics (torchmetrics.MetricCollection or None): Vectorized metrics for testing phase.
            val_vectorized_metrics (torchmetrics.MetricCollection or None): Vectorized metrics for validation phase.
        """
        super().__init__()
        self.model = model
        self.loss_fn = loss_fn

        self.train_metrics = metrics.clone(prefix="train/")
        self.val_metrics = metrics.clone(prefix="val/")
        self.test_metrics = metrics.clone(prefix="testing_phase/")

        self.scheduler_max_it = scheduler_max_it
        self.weight_decay = weight_decay
        self.lr = lr

        self.class_names = class_names

        self.test_vectorized_metrics = None
        self.val_vectorized_metrics = None
        if vectorized_metrics is not None:
            self.test_vectorized_metrics = vectorized_metrics.clone(
                prefix="testing_phase/"
            )
            self.val_vectorized_metrics = vectorized_metrics.clone(prefix="val/")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs a forward pass through the model.

        Args:
            x (torch.Tensor): Input tensor to the model.

        Returns:
            torch.Tensor: Output tensor produced by the model.
        """
        return self.model(x)

    def _common_step(self, batch, batch_idx):
        """
        Performs a common forward and loss computation step for a batch.

        Args:
            batch (Tuple[Tensor, Tensor]): A tuple containing input images `x` of shape [B, C, H, W] and target masks `y` of shape [B, H, W].
            batch_idx (int): Index of the current batch.

        Returns:
            Tuple[Tensor, Tensor, Tensor]: A tuple containing the ground truth masks `y`, the model predictions `y_hat`, and the computed loss.
        """
        x, y = batch
        y_hat = self(x)

        loss = self.loss_fn(y_hat, y)
        return y, y_hat, loss

    def training_step(self, batch, batch_idx):
        """
        Performs a single training step.

        Args:
            batch (Any): A batch of data containing input features and target labels.
            batch_idx (int): Index of the current batch.

        Returns:
            dict: A dictionary containing the computed loss for the current batch.

        Side Effects:
            - Updates training metrics with model predictions and ground truth.
            - Logs the training loss for the current epoch.
        """
        # Forward pass
        y, y_hat, loss = self._common_step(batch, batch_idx)

        self.train_metrics.update(y_hat, y)

        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return {"loss": loss}

    def on_train_epoch_end(self):
        """
        Called at the end of the training epoch.
        Logs the computed training metrics for the current epoch and resets the metric state for the next epoch.
        """

        self.log_dict(self.train_metrics.compute(), on_step=False, on_epoch=True)

        self.train_metrics.reset()

    def validation_step(self, batch, batch_idx):
        """
        Performs a single validation step during model evaluation.

        Args:
            batch (Tuple[Tensor, Tensor]): A tuple containing the input data and corresponding labels for the current batch.
            batch_idx (int): The index of the current batch.

        Returns:
            dict: A dictionary containing the validation loss for the current batch.

        Side Effects:
            - Updates validation metrics (`self.val_metrics` and optionally `self.val_vectorized_metrics`) with predictions and targets.
            - Logs the validation loss to the progress bar and for epoch-level aggregation.
        """
        # Forward pass
        y, y_hat, loss = self._common_step(batch, batch_idx)

        self.val_metrics.update(y_hat, y)

        if self.val_vectorized_metrics is not None:
            self.val_vectorized_metrics.update(y_hat, y)

        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return {"loss": loss}

    def apply_colormap(self, saliency_map, cmap="inferno"):
        """
        Applies a matplotlib colormap to a normalized saliency map and returns a heatmap.
        Args:
            saliency_map (np.ndarray): The input saliency map as a 2D numpy array.
            cmap (str, optional): The name of the matplotlib colormap to use. Defaults to "inferno".
        Returns:
            np.ndarray: The resulting heatmap as a uint8 numpy array with shape (H, W, 3).
        """

        saliency_map = (saliency_map - saliency_map.min()) / (
            saliency_map.max() - saliency_map.min() + 1e-8
        )
        heatmap = plt.get_cmap(cmap)(saliency_map)[:, :, :3]  # Remove alpha channel
        heatmap = (heatmap * 255).astype(np.uint8)
        return heatmap

    def log_saliency_maps(
        self,
        model,
        mask,
        input_image,
        saliency_fn,
        class_names=None,
        cmap="inferno",
        logger=None,
        step=None,
    ):
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
        input_np = (input_np - input_np.min()) / (
            input_np.max() - input_np.min() + 1e-8
        )

        saliency_images = []

        for cls in range(num_classes):
            binary_mask = output.argmax(dim=1) == cls

            sal_map = saliency_fn(
                model, input_image, mask=binary_mask, target_class=cls
            )  # expected: [H, W] numpy

            # Overlay on original image
            overlay = input_np.copy()
            heatmap = plt.get_cmap(cmap)(sal_map)[:, :, :3]  # RGB heatmap (no alpha)
            overlay = 0.3 * overlay + 0.7 * heatmap  # Blend original and saliency

            overlay = np.clip(overlay, 0, 1)
            label = class_names[cls] if class_names else f"Class {cls}"
            saliency_images.append(
                wandb.Image(overlay, caption=f"Saliency map overlayed: {label}")
            )

        logger.experiment.log(
            {
                "Input Image": [
                    wandb.Image(input_np, caption="Input Image"),
                    wandb.Image(
                        class_to_rgb(
                            mask.permute(1, 2, 0).squeeze(-1).detach().cpu().numpy()
                        ),
                        caption="Mask",
                    ),
                ],
                "Output Image": wandb.Image(
                    class_to_rgb(output.argmax(dim=1)[0].detach().cpu().numpy()),
                    caption="Output Image",
                ),
                "Saliency Maps": saliency_images,
                "epoch": step,
            }
        )

    def log_feature_ablation_maps(
        self,
        model,
        input_image,
        feature_ablation_fn,
        mask,
        class_names=None,
        cmap="inferno",
        logger=None,
        step=None,
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
        input_np = (input_np - input_np.min()) / (
            input_np.max() - input_np.min() + 1e-8
        )

        num_classes = model(input_image).shape[1]
        ablation_images = []

        for cls in range(num_classes):
            sal_map = feature_ablation_fn(
                model, input_image, cls, mask
            )  # <- renamed arg

            heatmap = self.apply_colormap(sal_map, cmap=cmap)
            label = class_names[cls] if class_names else f"Class {cls}"

            ablation_images.append(wandb.Image(heatmap, caption=f"Ablation: {label}"))

        logger.experiment.log(
            {
                "Input Image (Ablation)": wandb.Image(input_np, caption="Input Image"),
                "Feature Ablation Maps": ablation_images,
                "epoch": step,
            }
        )

    def on_validation_epoch_end(self):
        """
        Called at the end of the validation epoch.

        - Logs validation metrics using `self.val_metrics` and resets them.
        - If a datamodule is present, retrieves a sample batch from the validation dataloader,
          extracts a specific input image and mask, and logs saliency maps for visualization.
        - If vectorized validation metrics are available, computes and logs both scalar and
          per-class metrics to Weights & Biases (wandb), including visual bar plots for per-class metrics.
        - Resets vectorized validation metrics after logging.

        This method is typically used in PyTorch Lightning modules to handle custom logging,
        visualization, and metric tracking at the end of each validation epoch.
        """
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

        if self.val_vectorized_metrics is not None:
            results = self.val_vectorized_metrics.compute()
            for metric_name, metric_tensor in results.items():
                if metric_tensor.ndim == 0:
                    # Scalar metric
                    wandb.log(
                        {metric_name: metric_tensor.item(), "epoch": self.current_epoch}
                    )
                else:
                    # Build wandb.Table
                    table = wandb.Table(columns=["Class", metric_name])
                    for i, val in enumerate(metric_tensor):
                        table.add_data(self.class_names[i], val.item())

                    wandb.log(
                        {
                            f"{metric_name}": wandb.plot.bar(
                                table, "Class", metric_name, title=metric_name
                            ),
                            "epoch": self.current_epoch,
                        }
                    )

            self.val_vectorized_metrics.reset()

    def test_fps(self, device=None) -> float:
        """
        Measures the inference speed of the model in frames per second (FPS) using a dummy input.

        Args:
            device (str or torch.device, optional): The device on which to run the model for testing.
                If None, uses the model's current device.

        Returns:
            float: The number of frames per second (FPS) the model can process.

        Notes:
            - Performs a warm-up phase to stabilize GPU performance.
            - Uses a fixed input size of (1, 3, 224, 224).
            - Synchronizes CUDA operations to ensure accurate timing.
        """
        self.model.eval()
        if device is None:
            example_input = torch.randn(1, 3, 224, 224).to(self.device)
        else:
            self.model.to(torch.device(device))
            example_input = torch.randn(1, 3, 224, 224).to(torch.device(device))

        # Warm-up GPU
        for _ in range(10):
            with torch.no_grad():
                _ = self.model(example_input)

        # Measure over multiple iterations
        num_iterations = 100
        torch.cuda.synchronize()
        start_time = perf_counter()

        with torch.no_grad():
            for _ in range(num_iterations):
                _ = self.model(example_input)

        torch.cuda.synchronize()
        end_time = perf_counter()

        # Calculate FPS
        total_time = end_time - start_time
        fps = num_iterations / total_time

        return fps

    def test_step(self, batch, batch_idx):
        """
        Performs a single test step during model evaluation.

        Args:
            batch (Any): A batch of data from the test DataLoader.
            batch_idx (int): Index of the current batch.

        Returns:
            dict: A dictionary containing the loss value for the current batch.

        This method executes a forward pass using the provided batch, computes the loss,
        updates test metrics (including optional vectorized metrics), and logs the loss
        for the testing phase.
        """
        # Forward pass
        y, y_hat, loss = self._common_step(batch, batch_idx)

        self.test_metrics.update(y_hat, y)

        if self.test_vectorized_metrics is not None:
            self.test_vectorized_metrics.update(y_hat, y)

        self.log(
            "testing_phase/loss", loss, on_step=False, on_epoch=True, prog_bar=True
        )
        return {"loss": loss}

    def on_test_epoch_end(self):
        """
        Called at the end of the test epoch to log and reset test metrics.
        - Logs aggregated test metrics using `self.log_dict`.
        - Resets the test metrics accumulator.
        - Logs the frames-per-second (FPS) for the testing phase to Weights & Biases (wandb).
        - If vectorized test metrics are available:
            - Computes and logs each metric.
            - For scalar metrics, logs the value directly to wandb.
            - For vector (per-class) metrics, creates a wandb.Table and logs a bar plot for visualization.
            - Resets the vectorized test metrics accumulator.
        """

        self.log_dict(self.test_metrics.compute(), on_step=False, on_epoch=True)
        self.test_metrics.reset()
        wandb.log({"testing_phase/fps": self.test_fps()})

        if self.test_vectorized_metrics is not None:
            results = (
                self.test_vectorized_metrics.compute()
            )  # Dict: metric_name → Tensor
            for metric_name, metric_tensor in results.items():
                # metric_tensor might be scalar (e.g. macro accuracy) or vector (per-class)
                if metric_tensor.ndim == 0:
                    # Scalar metric
                    wandb.log(
                        {metric_name: metric_tensor.item(), "epoch": self.current_epoch}
                    )
                else:
                    # Build wandb.Table
                    table = wandb.Table(columns=["Class", metric_name])
                    for i, val in enumerate(metric_tensor):
                        table.add_data(f"Class {i}", val.item())

                    wandb.log(
                        {
                            f"{metric_name}": wandb.plot.bar(
                                table, "Class", metric_name, title=metric_name
                            ),
                            "epoch": self.current_epoch,
                        }
                    )

            self.test_vectorized_metrics.reset()

    def configure_optimizers(self):
        """
        Sets up the optimizer and learning rate scheduler for model training.
        Returns:
            Tuple[List[torch.optim.Optimizer], List[torch.optim.lr_scheduler._LRScheduler]]:
                - A list containing the AdamW optimizer initialized with the model's parameters, learning rate, and weight decay.
                - A list containing the CosineAnnealingLR scheduler configured with the optimizer and the maximum number of iterations.
        """

        optimizer = AdamW(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        scheduler = CosineAnnealingLR(optimizer, T_max=self.scheduler_max_it)
        return [optimizer], [scheduler]
