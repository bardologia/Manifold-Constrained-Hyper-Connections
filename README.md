# mHC: Manifold-Constrained Hyper-Connections (PyTorch)

This repository contains a lightweight PyTorch implementation of **mHC (Manifold-Constrained Hyper-Connections)** based on:

Xie et al., *mHC: Manifold-Constrained Hyper-Connections*, DeepSeek-AI, 2025.

## Overview

`mHC.py` implements an `mHC` module that:

- Computes dynamic pre/post/residual connection mappings from the current residual stream state.
- Constrains residual transport with Sinkhorn-Knopp normalization (approximately doubly-stochastic matrices).
- Aggregates multiple streams into a functional block input and redistributes the block output back to streams.

Expected input shape:

- `x`: `(batch_size, n_streams, d_model)`

## Requirements

- Python 3.9+
- PyTorch 2.0+

## Citation

If you use this implementation, please cite the original paper:

```text
Zhenda Xie, Yixuan Wei, Huanqi Cao, Chenggang Zhao, Chengqi Deng, Jiashi Li,
Damai Dai, Huazuo Gao, Jiang Chang, Kuai Yu, Liang Zhao, Shangyan Zhou,
Zhean Xu, Zhengyan Zhang, Wangding Zeng, Shengding Hu, Yuqing Wang,
Jingyang Yuan, Lean Wang, Wenfeng Liang.
mHC: Manifold-Constrained Hyper-Connections.
arXiv:2512.24880, 2025.
```

BibTeX:

```bibtex
@article{xie2025mhc,
  title={mHC: Manifold-Constrained Hyper-Connections},
  author={Xie, Zhenda and Wei, Yixuan and Cao, Huanqi and Zhao, Chenggang and Deng, Chengqi and Li, Jiashi and Dai, Damai and Gao, Huazuo and Chang, Jiang and Yu, Kuai and Zhao, Liang and Zhou, Shangyan and Xu, Zhean and Zhang, Zhengyan and Zeng, Wangding and Hu, Shengding and Wang, Yuqing and Yuan, Jingyang and Wang, Lean and Liang, Wenfeng},
  journal={arXiv preprint arXiv:2512.24880},
  year={2025}
}
```

## Notes

This repository is an independent implementation intended for research and educational use.
