<h2 align="center">Temporal Smoothness-Aware Rate-Distortion Optimized 4D Gaussian Splatting</h2>
<p align="center">
  <strong>Hyeongmin Lee</strong>
  &nbsp;·&nbsp;
  <strong>Kyungjune Baek</strong>
  <br>
  NeurIPS 2025
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2507.17336"><strong><code>Paper (arXiv)</code></strong></a>
  &nbsp;
  <a href="https://github.com/HyeongminLEE/RD4DGS"><strong><code>Source Code</code></strong></a>
</p>

<div align='center'>
  <br>
  <!-- TODO: Add thumbnail/GIF here -->
  <img src="img/placeholder.png" width=70%>
  <br>
  <em>TBD: Result visualization will be added.</em>
</div>

<br>

## Summary

This repository provides the official implementation of **RD4DGS**, an end-to-end **Rate-Distortion optimized compression framework** for **4D Gaussian Splatting**. Built upon [Ex4DGS](https://github.com/juno181/Ex4DGS), our method achieves up to **91× compression** while maintaining high visual fidelity for dynamic scene rendering.

### Key Features
- **Adaptive Pruning**: Learnable masks with Straight-Through Estimator (STE) for Gaussian and SH coefficient pruning
- **Entropy-Constrained Vector Quantization (ECVQ)**: Applied to SH, rotation, and scale parameters
- **Haar Wavelet Transform**: Exploits temporal smoothness in dynamic position trajectories for efficient compression

<br>

## Contents

1. [Setup](#setup)
2. [Dataset Preparation](#dataset-preparation)
3. [Training](#training)
4. [Evaluation](#evaluation)
5. [Pretrained Models](#pretrained-models)
6. [Citation](#citation)
7. [Acknowledgements](#acknowledgements)

<br>

## Setup

### Environment Setup

Clone the repository:
```shell
git clone https://github.com/HyeongminLEE/RD4DGS.git
cd RD4DGS
```

Create and activate the conda environment:
```shell
conda create -n rd4dgs python=3.9
conda activate rd4dgs
```

Install PyTorch (ensure compatibility with your CUDA version). We tested with PyTorch 2.1.2 and CUDA 11.8 on RTX 4090:
```shell
# Example for PyTorch 2.1.2 with CUDA 11.8
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118
```

Install the remaining dependencies:
```shell
pip install -r requirements.txt
```

Install the custom CUDA extensions:
```shell
pip install submodules/diff_gaussian_rasterization_df
pip install submodules/simple-knn
pip install submodules/imcl-compression
```

<br>

## Dataset Preparation

We use the **Neural 3D Video (N3V)** and **Technicolor** datasets. For dataset download and preprocessing, please follow the instructions in [Ex4DGS](https://github.com/juno181/Ex4DGS).

<br>

## Training

The training consists of two stages: **pretraining** and **RD optimization fine-tuning**.

### 1. Pretraining

First, pretrain the model without rate-distortion optimization:
```shell
python train.py \
    --config configs/N3V/n3v_rdo_pretrain.json \
    --model_path output/pretrain/n3v/<scene_name> \
    --source_path <path_to_dataset>/<scene_name> \
    --no_rdo
```

### 2. RD Optimization Fine-tuning

After pretraining, fine-tune with RD optimization. We provide 6 compression levels (`lev1` to `lev6`), where higher levels indicate stronger compression.

```shell
python train.py \
    --config configs/N3V/n3v_rdo_wv_lev<level>.json \
    --model_path output/rdo/n3v/<scene_name>/lev<level> \
    --source_path <path_to_dataset>/<scene_name> \
    --start_checkpoint output/pretrain/n3v/<scene_name>/chkpnt30000.pth
```

**Example** (training `coffee_martini` with compression level 4):
```shell
# Pretraining
python train.py \
    --config configs/N3V/n3v_rdo_pretrain.json \
    --model_path output/pretrain/n3v/coffee_martini \
    --source_path ./data/n3v/coffee_martini \
    --no_rdo

# RD Fine-tuning (lev4)
python train.py \
    --config configs/N3V/n3v_rdo_wv_lev4.json \
    --model_path output/rdo/n3v/coffee_martini/lev4 \
    --source_path ./data/n3v/coffee_martini \
    --start_checkpoint output/pretrain/n3v/coffee_martini/chkpnt30000.pth
```

### Technicolor Dataset

For Technicolor, use the scene-specific configs:
```shell
python train.py \
    --config configs/Technicolor/<Scene>_rdo_wv_lev<level>.json \
    --model_path output/rdo/technicolor/<scene_name>/lev<level> \
    --source_path <path_to_dataset>/<scene_name> \
    --start_checkpoint output/pretrain/technicolor/<scene_name>/chkpnt30000.pth
```

<br>

## Evaluation

To evaluate a trained model:
```shell
python validation.py \
    --model_path <path_to_trained_model> \
    --source_path <path_to_dataset> \
    --iteration 40000 \
    --skip_train
```

Add `--save_img` to save rendered images.

<br>

## Pretrained Models

<!-- TODO: Add pretrained model download links -->
**TBD**: Pretrained models will be released soon.

<br>

## Citation

If you find this work useful, please cite our paper:

```bibtex
@inproceedings{lee2025rd4dgs,
  title={Temporal Smoothness-Aware Rate-Distortion Optimized 4D Gaussian Splatting},
  author={Lee, Hyeongmin and Baek, Kyungjune},
  booktitle={Advances in Neural Information Processing Systems},
  year={2025}
}
```

<br>

## Acknowledgements

This project builds upon the excellent work of:
- [Ex4DGS](https://github.com/juno181/Ex4DGS) - Fully Explicit Dynamic Gaussian Splatting (NeurIPS 2024)
- [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting) - 3D Gaussian Splatting for Real-Time Radiance Field Rendering (SIGGRAPH 2023)

We thank the authors for releasing their code.

<br>

## License

This project is licensed under the [CC BY-NC 4.0 License](LICENSE.txt).
