# Lab 5: Diffusion Models for Full Waveform Inversion (FWI)

This lab introduces **conditional diffusion models** for generating velocity models in Full Waveform Inversion (FWI) applications.

## Overview

Full Waveform Inversion is a technique used in geophysics to create high-resolution images of the Earth's subsurface. This lab demonstrates how diffusion models can be used to generate realistic velocity models conditioned on seismic observations.

## Structure

```
lab_5/
├── diffusion_fwi_generation_notebook.ipynb   # Main tutorial notebook
├── checkpoints/                               # Pre-trained model weights
│   └── conditional/
├── outputs_cond/                              # Sample outputs
├── utils/                                     # Utility functions
│   ├── diffusion.py                          # Diffusion process utilities
│   ├── nn.py                                 # Neural network components
│   ├── preconditioning.py                    # Preconditioning utilities
│   ├── metrics.py                            # Evaluation metrics
│   └── plot.py                               # Visualization utilities
└── requirements.txt
```

## Notebook

| Notebook | Description |
|----------|-------------|
| `diffusion_fwi_generation_notebook.ipynb` | Generate velocity models using conditional diffusion |

## Key Concepts

### Diffusion Models
- **Forward process**: Gradually add noise to data
- **Reverse process**: Learn to denoise and generate samples
- **Conditioning**: Guide generation based on input observations

### Full Waveform Inversion
- Estimate subsurface velocity from seismic data
- Traditional methods are computationally expensive
- Diffusion models can generate plausible velocity models quickly

## Prerequisites

```bash
pip install -r requirements.txt
```

## References

- [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239)
- [Score-Based Generative Modeling](https://arxiv.org/abs/2011.13456)
