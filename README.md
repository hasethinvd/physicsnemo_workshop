# PhysicsNeMo workshop

This workshop will provide researchers hands-on approaches on how to use NVIDIA PhysicsNeMo, a framework that combines physics and partial differential equations (PDEs) with artificial intelligence (AI) to build robust models. Participants will learn about the PhyicsNeMo sym utilities to infuse physics during training and inference stage.

## Workshop contents:

The content is structured in two modules covering the following: 


- Lab 1 : [Physics Informed FNO for Nonlinear Shallow Water Equations](./lab_1/swe_nonlinear_pino.ipynb)

    This example demonstrates physics informing of a data-driven model using numerical derivatives (PINO) during the training stage.
  
- Lab 2 : [Learning the flow field of Stokes flow](./lab_2/stokes_mgn.ipynb)

    This example demonstrates how to train the MeshGraphNet model to learn the flow field of Stokes flow and further improve the accuary of the model predictions by physics-informed inference. This example also demonstrates how to use physics utilites from PhysicsNeMo-Sym to introduce physics-based constraints.

  - Part 1: Physics-Informed inference using PINNs

  - Part 2: Physics-Informed inference using Meshgraphnet finetuning