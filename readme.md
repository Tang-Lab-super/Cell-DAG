# DAGAST: Dual Attention-based Graph Autoencoder for Spatial Trajectory Inference
## Overview

**DAGAST** is an interpretable deep learning model that is specifically designed to identify patterns of continuous changes in cell states in spatial transcriptomes, such as developmental differentiation trajectories. It can also infer **functional genesets** and **functional spatial domains** that are closely related to the change process, thereby providing new ideas for understanding the mechanism of cell state changes in tissue space.

<p align = "center"><img src="./Tutorial/figs/Flowchart.png" width="500" /></p>

---

## Tutorial

We provide examples of using DAGAST in three classic spatial transcriptome data platforms.

[Tutorial 1: Application on SeqFISH mouse embryo dataset.](./Tutorial/SeqFISH-pipeline.md)  
[Tutorial 2: Application on Stereo-seq axolotl telencephalon brain dataset.](./Tutorial/Stereo-seq-pipeline.md)  
[Tutorial 3: Application on 10x Visium traumatic brain injury (TBI) dataset.](./Tutorial/Visium-pipeline.md)

---

## Installation

We use Python 3.8 in the conda environment. The versions of the main dependencies are shown in [requirements.txt](./requirements.txt).


## Improvements

We welcome any comments about DAGAST, and if you find bugs or have any ideas, feel free to leave a comment FAQ. DAGAST doesn't fully test on macOS.


