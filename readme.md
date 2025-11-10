# DAGAST: Dual Attention-based Graph Autoencoder for Spatial Trajectory Inference
## Overview

**DAGAST** is an interpretable deep learning model that is specifically designed to identify patterns of continuous changes in cell states in spatial transcriptomes, such as developmental differentiation trajectories. It can also infer **functional genesets** and **functional spatial domains** that are closely related to the change process, thereby providing new ideas for understanding the mechanism of cell state changes in tissue space.

<p align = "center"><img src="./Tutorial/figs/Flowchart.png" width="500" /></p>

---

## Tutorial

We obtained high-quality real spatial transcriptomics datasets covering three sequencing platforms from public databases and demonstrate the capability of DAGAST in spatial trajectory inference and spatial pseudotime calculation.

[Tutorial 1: Application on the SeqFISH dataset of early mouse embryonic development (GSE197353) (Sampath Kumar et al., 2023).](./Tutorial/SeqFISH-pipeline.md)  
[Tutorial 2: Application on the Stereo-seq dataset of axolotl brain regeneration (CNP0002068) (Wei et al., 2022).](./Tutorial/Stereo-seq-pipeline.md)  
[Tutorial 3: Application on the Visium HD dataset of the mouse cerebral cortex (10x genomics).](./Tutorial/Visium-pipeline.md)

---

## Installation

We use Python 3.8 in the conda environment. The versions of the main dependencies are shown in [requirements.txt](./requirements.txt).

## Reference and Citation

Development  and application of a dual-branch mechanism-based algorithm for inferring spatial differentiation trajectories of cells in tissues

## Improvements

We welcome any comments about DAGAST, and if you find bugs or have any ideas, feel free to leave a comment FAQ. DAGAST doesn't fully test on macOS.






