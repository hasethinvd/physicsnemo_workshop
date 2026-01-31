# PhysicsNeMo workshop

This workshop will provide researchers hands-on approaches on how to use NVIDIA PhysicsNeMo, a framework that combines physics and partial differential equations (PDEs) with artificial intelligence (AI) to build robust models. Participants will learn about the PhyicsNeMo sym utilities to infuse physics during training and inference stage.

## Workshop contents:

- **Lab 1**: [Physics Informed FNO for Nonlinear Shallow Water Equations](./lab_1/swe_nonlinear_pino.ipynb)

    Physics informing of a data-driven model using numerical derivatives (PINO) during training.
  
- **Lab 2**: [Learning the flow field of Stokes flow](./lab_2/stokes_mgn.ipynb)

    Train MeshGraphNet to learn Stokes flow and improve accuracy via physics-informed inference.

- **Lab 3**: [Transolver - Physics-Aware Transformers for PDEs](./lab_3/README.md)

    Introduction to Transolver and Physics-Attention for efficient transformer-based PDE solvers.

- **Lab 4**: [Reservoir Simulation with xMGN](./lab_4/xmgn/README.md)

    Multi-scale graph networks for reservoir simulation.

- **Lab 5**: [Diffusion Models for FWI](./lab_5/diffusion_fwi_generation_notebook.ipynb)

    Conditional diffusion models for full waveform inversion.