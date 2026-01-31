# Lab 3: Transolver - Physics-Aware Transformers for PDEs

This lab introduces **Transolver**, a transformer architecture designed for physics simulations. Unlike standard transformers that have O(N²) complexity, Transolver uses **Physics-Attention** to achieve O(N) complexity while learning physically meaningful representations.

This lab follows the same structure as PhysicsNeMo examples (see `examples/cfd/darcy_transolver`).

## Structure

```
lab_3/
├── conf/
│   └── config.yaml              # Configuration (Hydra/OmegaConf)
├── 01_physics_attention.ipynb   # Theory: Standard vs Physics-Attention
├── 02_transolver_stokes.ipynb   # Training with visualization
├── train_transolver_stokes.py   # Production training script
├── utils.py                     # Utility functions
└── requirements.txt
```

## Notebooks

| Notebook | Description |
|----------|-------------|
| `01_physics_attention.ipynb` | Understand why standard attention is expensive for physics and how Physics-Attention solves it |
| `02_transolver_stokes.ipynb` | Train PhysicsNeMo's Transolver on Stokes flow and visualize learned slices |

## Training Script

For production training (following PhysicsNeMo patterns):

```bash
python train_transolver_stokes.py
```

## Key Concepts

### The Problem with Standard Attention
- Standard transformers compute attention between ALL pairs of points: O(N²)
- For meshes with 100K+ points, this is computationally prohibitive
- But physics is mostly **local** - distant points rarely interact directly

### The Transolver Solution: Physics-Attention
1. **Slice**: Learn to group mesh points into M physics-meaningful "slices"
2. **Aggregate**: Compress N points → M tokens (N >> M)
3. **Attend**: Standard attention on M tokens only: O(M²) where M ≈ 64
4. **Deslice**: Broadcast back to N points

This reduces complexity from O(N²) to O(N·M) ≈ O(N).

## Prerequisites

```bash
pip install -r requirements.txt
```

## References

- [Transolver: A Fast Transformer Solver for PDEs on General Geometries](https://arxiv.org/abs/2402.02366)
- [Learning Mesh-Based Simulation with Graph Networks](https://arxiv.org/abs/2010.03409)
