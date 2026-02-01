# PhysicsNeMo Workshop

This workshop provides hands-on approaches on how to use NVIDIA PhysicsNeMo, a framework that combines physics and partial differential equations (PDEs) with artificial intelligence (AI) to build robust models. Participants will learn about the PhysicsNeMo sym utilities to infuse physics during training and inference.

## Quick Start with Docker

### Prerequisites
- Docker with NVIDIA Container Toolkit
- NVIDIA GPU with CUDA support

### Option 1: Docker Compose (Recommended)
```bash
# Clone the repository
git clone https://github.com/hasethinvd/physicsnemo_workshop.git
cd physicsnemo_workshop

# Build and start the container
docker compose up --build

# Access Jupyter Lab at http://localhost:8888
```

### Option 2: Docker CLI
```bash
# Build the image
docker build -t physicsnemo-workshop .

# Run with GPU support
docker run --gpus all -it \
    -p 8888:8888 \
    -v $(pwd):/workspace/physicsnemo_workshop \
    --shm-size=16gb \
    physicsnemo-workshop

# Access Jupyter Lab at http://localhost:8888
```

### Running Specific Labs
```bash
# Enter the container shell
docker compose exec workshop bash

# Run Lab 1 (PINO)
cd lab_1 && python train_swe_nl_pino.py

# Run Lab 2 (Transolver)
cd lab_2 && python train_transolver_stokes.py

# Run Lab 3 (xMGN)
cd lab_3/xmgn && python src/train.py

# Run Lab 4 (MeshGraphNet)
cd lab_4 && python train.py

# Run Lab 5 (Diffusion)
cd lab_5 && jupyter notebook diffusion_fwi_generation_notebook.ipynb
```

---

## Workshop Contents

- **Lab 1**: [Physics Informed FNO for Nonlinear Shallow Water Equations](./lab_1/swe_nonlinear_pino.ipynb)

    Physics informing of a data-driven model using numerical derivatives (PINO) during training.
  
- **Lab 2**: [Transolver - Physics-Aware Transformers for PDEs](./lab_2/README.md)

    Introduction to Transolver and Physics-Attention for efficient transformer-based PDE solvers.

- **Lab 3**: [Reservoir Simulation with xMGN](./lab_3/xmgn/README.md)

    Multi-scale graph networks for reservoir simulation.

- **Lab 4**: [Learning the flow field of Stokes flow](./lab_4/stokes_mgn.ipynb)

    Train MeshGraphNet to learn Stokes flow and improve accuracy via physics-informed inference.

- **Lab 5**: [Diffusion Models for FWI](./lab_5/diffusion_fwi_generation_notebook.ipynb)

    Conditional diffusion models for full waveform inversion.