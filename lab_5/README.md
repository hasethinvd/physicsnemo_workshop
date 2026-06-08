# Lab 5: Diffusion Models for Full Waveform Inversion (FWI)

This lab is based on the [diffusion_fwi](https://github.com/NVIDIA/physicsnemo/tree/main/examples/geophysics/diffusion_fwi)
example from [PhysicsNeMo](https://github.com/NVIDIA/physicsnemo) (v2.1.0+).

## Problem Overview

Full Waveform Inversion (FWI) is a seismic imaging technique that reconstructs subsurface
velocity models from recorded seismic data. This lab demonstrates how diffusion models can
generate realistic velocity models conditioned on seismic observations — providing fast,
probabilistic alternatives to classical gradient-based inversion.

Key application areas:
- Hydrocarbon exploration (guiding drilling decisions)
- CO₂ storage monitoring
- Global and regional seismology

## Prerequisites

### Install dependencies

```bash
pip install -r lab_5/requirements.txt
```

Or for the full workshop:

```bash
pip install -r requirements.txt
```

> **Note:** `nvidia-physicsnemo[sym]>=2.1.0` requires `torch>=2.10.0`.
> The `[sym]` extra adds `sympy` support for symbolic PDE definitions.

### Optional: deepwave for physics-informed guidance

```bash
pip install "deepwave>=0.0.21"
```

## Structure

```
lab_5/
├── diffusion_fwi_generation_notebook.ipynb   # Main tutorial notebook
├── checkpoints/                               # Pre-trained model weights
│   └── conditional/
│       └── DiffusionFWINet.0.980.mdlus       # Checkpoint for inference
├── outputs_cond/                              # Pre-generated sample outputs
├── utils/                                     # Utility modules
│   ├── diffusion.py                          # Diffusion sampler and guidance
│   ├── nn.py                                 # DiffusionFWINet model definition
│   ├── preconditioning.py                    # EDM preconditioning
│   ├── metrics.py                            # Evaluation metrics
│   └── plot.py                               # Visualization helpers
└── requirements.txt
```

## Running the Notebook

```bash
cd lab_5
jupyter lab diffusion_fwi_generation_notebook.ipynb
```

The notebook has two modes:

1. **Visualization only** — loads pre-generated results from `outputs_cond/` without
   any model or GPU required. Run the standalone visualization cell at the top.

2. **Full inference** — runs the conditional diffusion model on new samples.
   Requires a CUDA GPU and the pre-trained checkpoint in `checkpoints/conditional/`.

## Key Concepts

### Diffusion Models
- **Forward process**: Gradually corrupt data with noise
- **Reverse process**: Learn to denoise toward realistic samples
- **Conditioning**: Steer generation using seismic observations

### PhysicsNeMo APIs Used
| Import | Purpose |
|--------|---------|
| `physicsnemo.Module` | Base class for saveable/loadable models |
| `physicsnemo.diffusion.utils.StackedRandomGenerator` | Reproducible batched sampling |
| `physicsnemo.models.diffusion_unets.SongUNetPosEmbd` | Score network architecture |
| `physicsnemo.core.meta.ModelMetaData` | Model metadata for checkpointing |

## Upstream Example

The full training script and dataset preparation scripts are available in the
upstream PhysicsNeMo repository:

```
examples/geophysics/diffusion_fwi/
├── train.py          # Training script
├── generate.py       # Generation / inference script
├── conf/             # Hydra configuration files
├── datasets/         # Dataset preparation utilities
└── README.md         # Detailed problem description and training instructions
```

See the [upstream README](https://github.com/NVIDIA/physicsnemo/blob/main/examples/geophysics/diffusion_fwi/README.md)
for the full problem description and training details.

## References

- [Score-Based Generative Modeling through SDEs](https://arxiv.org/abs/2011.13456) (Song et al., 2021)
- [Elucidating the Design Space of Diffusion-Based Generative Models](https://arxiv.org/abs/2206.00364) (Karras et al., 2022)
- [PhysicsNeMo diffusion_fwi example](https://github.com/NVIDIA/physicsnemo/tree/main/examples/geophysics/diffusion_fwi)
