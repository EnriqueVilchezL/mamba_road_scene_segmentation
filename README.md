# U-Net Variants for Semantic Segmentation in Autonomous Driving

This project evaluates the performance of three semantic segmentation models — **U-Net**, **Swin U-Net**, and **Mamba U-Net** — in the context of **autonomous driving** using the **BDD100K semantic segmentation dataset**. The primary focus of this study is on **balanced accuracy** due to the **severe class imbalance** in the dataset.

---

## 🚗 Dataset

We use the **BDD100K Semantic Segmentation Subset**, which consists of:

- 7,000 training images  
- 1,000 validation images  
- Images labeled with 20 semantic classes relevant to urban driving scenes

### Class Label Map

The classes present in the dataset are:

```python
LABEL_MAP = {  
0: "road",  
1: "sidewalk",  
2: "building",  
3: "wall",  
4: "fence",  
5: "pole",  
6: "traffic light",  
7: "traffic sign",  
8: "vegetation",  
9: "terrain",  
10: "sky",  
11: "person",  
12: "rider",  
13: "car",  
14: "truck",  
15: "bus",  
16: "train",  
17: "motorcycle",  
18: "bicycle",  
19: "unknown"  
}
```

The dataset presents a long-tail distribution, with common classes like "road" and "sky" dominating over rare ones like "train", "motorcycle", or "rider".

---

## 🔍 Objective

The goal is to compare architectural performance in handling imbalanced multi-class segmentation, particularly with regard to:

- Balanced Accuracy
- Accuracy
- Mean IoU (mIoU)  
- Class-wise Precision & Recall  

---

## 🧠 Models Compared

| Model        | Architecture Highlights                                 |
|--------------|----------------------------------------------------------|
| U-Net        | Classic CNN-based encoder-decoder with skip connections |
| Swin U-Net   | Transformer-based encoder (Swin Transformer) with hierarchical representations |
| Mamba U-Net  | State Space Model-based encoder with long-range temporal dynamics |

All models were trained under the same experimental setup for fair comparison.

---

## ⚙️ Environment Setup

To reproduce the results, create a Conda environment using the provided YAML file:

```bash
conda env create -f environment_unet.yml  
conda activate unet_segmentation_env
```

All required libraries, including PyTorch, and model-specific dependencies, are included.

---

## 📈 Training Details

- Optimizer: AdamW  
- Scheduler: CosineAnnealingLR  
- Loss Function: Unified Focal Loss (to handle class imbalance)  
- Epochs: 100
- Initial Learning Rate: 3e-4 (decayed with cosine schedule)  
- Batch Size: 24

The specifics of the training for each model is detailed in [training.md](training.md)
---

## 📊 Results Summary

> The testing results are:

| Model        | Balanced Accuracy | mIoU | Notes |
|--------------|-------------------|------|-------|
| U-Net        | 0.64              | 0.52 | Strong on common classes, weak on rare |
| Swin U-Net   | 0.68              | 0.57 | Better generalization to rare classes |
| Mamba U-Net  | 0.71              | 0.59 | Best at capturing rare patterns due to long-range context |

---

## 📁 Directory Structure

```bash
.
├── environment_unet.yml      # Conda environment definition  
├── data/                     # BDD100K Semantic Segmentation Dataset
├── src/                      # Code for experiments and models
    ├── models/                   # U-Net, Swin U-Net, Mamba U-Net architectures
        ├── unet.py                   # U-Net model pytorch implementation
        ├── swin_unet.py              # Swin-Unet model pytorch implementation
        ├── mamba_unet.py             # Mamba-Unet model pytorch implementation
        └── lightning_model.py        # General pytorch lightning segmentation model implementation

    ├── experiments/              # Experimental setup for all models
        ├── e0_unet.py                # Experiment 0: Experimental setup for training and validating Unet
        ├── e1_swin_unet.py           # Experiment 1: Experimental setup for training and validating Swin-Unet
        └── e2_mamba_unet.py          # Experiment 2: Experimental setup for training and validating Mamba-Unet

    ├── configuration.py          # Experimental hyperparameters and utility variables to train the models
    ├── data_loader.py            # Datamodules and dataloaders to load the training and validation data
    ├── explainability.py         # Complementary functions to calculate explicability mechanisms such as Saliency
    ├── loss.py                   # Unified Focal Loss pytorch implementation
    └── utils.py                  # Special functions to get and set torch specifics

└── README.md  
```

---

## 🚀 Future Work

- Test model's suitability for real time
- Extend comparison to video segmentation tasks  
- Explore hybrid architectures (e.g., Swin + Mamba fusion)  
- Explore other datasets

---

## 📜 License

This project is released under the MIT License.

---

## 🤝 Acknowledgements

### Dataset

- BDD100K Dataset: https://www.kaggle.com/datasets/solesensei/solesensei_bdd100k

### Code

- Swin-Unet code and architecture: https://github.com/microsoft/Swin-Transformer  
- Mamba-Unet code: https://github.com/ziyangwang007/Mamba-UNet 

### Architectures

- Unet architecture: https://arxiv.org/abs/1505.04597
- Swin-Unet architecture: https://arxiv.org/abs/2105.05537
- Mamba-Unet architecture: https://arxiv.org/abs/2402.05079
