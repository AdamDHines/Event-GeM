# :gem: Event-GeM: Multi-Scale Time Surfaces with 2D-Topology for Event-Based Visual Place Recognition
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)](https://creativecommons.org/licenses/by-nc-sa/4.0/)
[![Pixi Badge](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/prefix-dev/pixi/main/assets/badge/v0.json)](https://pixi.sh)
[![stars](https://img.shields.io/github/stars/AdamDHines/Event-GeM.svg?style=flat-square)](https://github.com/AdamDHines/Event-GeM/stargazers)
[![GitHub repo size](https://img.shields.io/github/repo-size/AdamDHines/Event-GeM.svg?style=flat-square)](./README.md)

This repository contains the code for Event-GeM — an event-based visual place recognition (VPR) pipeline that uses a pre-trained Swin Transformer feature extractor with 2D homology for keypoint-based match re-ranking.

[space for schema image]

Event-GeM uses features from the [Event-Camera-Data-Pre-Training](https://github.com/Yan98/Event-Camera-Data-Pre-training) and a generalized mean (GeM) pooling layer to generate initial matches from event frames. [SuperEvent](https://github.com/ethz-mrl/SuperEvent) then allows for 2D homology re-ranking of the TopK matches based on keypoint selection for improved recall performance. Datasets for VPR are managed and generated using [Event-LAB](https://github.com/EventLAB-Team/Event-LAB).

## Getting Started :rocket:
Event-GeM is powered by [pixi](https://pixi.sh/latest/) for all dependency and package management. If not already installed, run the following in your command terminal:

```terminal
curl -fsSL https://pixi.sh/install.sh | sh
```

_For more information, please see the [pixi documentation](https://pixi.sh/latest/)._

Once installed, you can quickly try Event-GeM with our demo by running the following in your command terminal:

```terminal
pixi run demo
```

_Please note: this will require **~35.9GB** of storage space to download and generate event data, ensure you have enough space on your local disk before proceeding._

## Feature Extraction

## Re-ranking and Recall@K Analysis

## Citation

## Contributing and Issues