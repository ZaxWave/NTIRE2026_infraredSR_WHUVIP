# 🛰️ QAHAT: Quality-Aware Hybrid Attention Transformer
### [NTIRE 2026 Challenge on Remote Sensing Infrared Image Super-Resolution (x4)](https://cvlai.net/ntire/2026/) @ [CVPR 2026](https://cvpr.thecvf.com/)

**Team Name:** WHU-VIP (Team ID: 11)  
**Authors:** Ce Wang, Xingwei Zhong, Wanjie Sun

[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](LICENSE)
[![NTIRE](https://img.shields.io/badge/CVPR-NTIRE_2026-blue?style=for-the-badge)](https://cvlai.net/ntire/2026/)

---

## 🌟 Overview
This repository contains the official implementation of **QAHAT**, the solution developed by team **WHU-VIP** for the NTIRE 2026 Infrared Image Super-Resolution challenge.

### Key Features:
- **QEM (Quality Estimation Module):** Dynamically assesses input quality to guide the restoration process.
- **AFB (Adaptive Fusion Block):** Efficiently integrates spatial and channel-wise attention.
- **SQA (Stacked Quality Attention):** Captures long-range dependencies tailored for infrared remote sensing textures.
- **Optimized Loss:** Trained with a custom weighted loss: $\mathcal{L} = -PSNR - 20 \cdot SSIM$.

---

## 🛠️ Proposed Method
![QAHAT Architecture](./figs/QAHAT.png)  
*Figure 1: The overall architecture of our proposed Quality-Aware Hybrid Attention Transformer (QAHAT).*

---

## 📦 Installation
```bash
# Clone the repository
git clone [https://github.com/ZaxWave/NTIRE2026_infraredSR_WHUVIP.git](https://github.com/ZaxWave/NTIRE2026_infraredSR_WHUVIP.git)
cd NTIRE2026_infraredSR_WHUVIP

# Create environment
conda create -n qahat python=3.8 -y
conda activate qahat

# Install dependencies
pip install -r requirements.txt
