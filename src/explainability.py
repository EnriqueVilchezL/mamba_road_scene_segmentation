from captum.attr import Saliency, FeatureAblation
import torch
def get_segmentation_saliency(model, image, mask, target_class):
    """
    Compute saliency map for a given image and target_class in a segmentation model.

    Args:
        model (torch.nn.Module): Segmentation network outputting shape [B, C, H, W]
        image (torch.Tensor): Input image tensor shape [1, C, H, W], requires_grad=True
        target_class (int): Class index to compute saliency for

    Returns:
        np.ndarray: Normalized saliency heatmap shape [H, W]
    """
    model.eval()
    image = image.detach().clone().requires_grad_(True)
    def forward_fn(x):
        out = model(x)  # [1, C, H, W]
        # Sum pixel-logits for the target class
        logits = out[:, target_class, :, :]
        return (logits * mask).sum(dim=(1, 2))

    saliency = Saliency(forward_fn)
    attributions = saliency.attribute(image)  # [1, C, H, W]

    # Take absolute max across channels
    sal = attributions[0].abs().detach().cpu().numpy()
    if sal.ndim == 3:  # when C > 1
        sal = sal.max(axis=0)

    # Normalize to [0, 1]
    sal = (sal - sal.min()) / (sal.max() - sal.min() + 1e-8)
    return sal

def compute_feature_ablation(model, input_image, target_class, mask, cmap="inferno"):
    """
    Computes feature ablation saliency for a given input and target class.

    Args:
        model (torch.nn.Module): The segmentation model.
        input_image (Tensor): Image tensor of shape [1, C, H, W].
        target_class (int): Target class index for which saliency is computed.
        mask (Tensor): Feature mask from model prediction (e.g., torch.max).
        cmap (str): Matplotlib colormap to use.

    Returns:
        Tuple[np.ndarray, np.ndarray]: (input_image_np, heatmap_np) for logging or visualization.
    """
    model.eval()

    def agg_wrapper(input_tensor):
        output = model(input_tensor)
        return (output * (output.argmax(1, keepdim=True) == target_class).float()).sum(dim=(2, 3))

    ablator = FeatureAblation(agg_wrapper)

    ablation_attr = ablator.attribute(
        input_image,
        feature_mask=mask[0],
        perturbations_per_eval=1,
        target=target_class
    )

    # Normalize input
    input_np = input_image[0].permute(1, 2, 0).detach().cpu().numpy()
    input_np = (input_np - input_np.min()) / (input_np.max() - input_np.min() + 1e-8)

    # Normalize saliency
    attr_np = ablation_attr[0].cpu().permute(1, 2, 0).detach().numpy()
    attr_np = (attr_np - attr_np.min()) / (attr_np.max() - attr_np.min() + 1e-8)

    return attr_np