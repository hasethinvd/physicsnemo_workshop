# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
#
# PhysicsNeMo Workshop Dockerfile
# Builds container with all dependencies for Labs 1-5

ARG BASE_CONTAINER=nvcr.io/nvidia/pytorch:26.04-py3
FROM ${BASE_CONTAINER}

# Avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Set working directory
WORKDIR /workspace

# ============================================
# System dependencies
# ============================================
RUN apt-get update && apt-get install -y --no-install-recommends \
    git-lfs \
    graphviz \
    libgl1 \
    libglib2.0-0 \
    wget \
    curl \
    zip \
    unzip \
    && git lfs install \
    && rm -rf /var/lib/apt/lists/*

# ============================================
# Update pip and setuptools, add build tools
# ============================================
RUN pip install --upgrade pip setuptools wheel hatchling editables

# ============================================
# PhysicsNeMo (nvidia-physicsnemo v2.1.0)
# physicsnemo.sym is now built-in via the [sym] extra.
# The separate physicsnemo-sym package has been archived.
# ============================================
ARG PHYSICSNEMO_VERSION=2.1.0
RUN pip install --no-cache-dir "nvidia-physicsnemo[sym]==${PHYSICSNEMO_VERSION}"

# ============================================
# Core ML/Scientific dependencies
# ============================================
RUN pip install --no-cache-dir \
    "numpy>=1.22.4" \
    "scipy>=1.10.0" \
    "h5py>=3.15.0" \
    "matplotlib>=3.8.0" \
    "einops>=0.8.1"

# ============================================
# Hydra / Config management
# ============================================
RUN pip install --no-cache-dir \
    "hydra-core>=1.3.2" \
    "omegaconf>=2.3.0"

# ============================================
# Visualization
# ============================================
RUN pip install --no-cache-dir \
    "vtk>=9.2.6" \
    "pyvista>=0.43.0" \
    "plotly>=5.18.0" \
    "imageio>=2.31.0"

# ============================================
# Experiment tracking
# ============================================
RUN pip install --no-cache-dir \
    "wandb>=0.15.1" \
    "mlflow>=2.1.1" \
    "termcolor>=2.1.1" \
    "tqdm>=4.60.0"

# ============================================
# Lab 3: xMGN (PyTorch Geometric)
# ============================================
RUN pip install --no-cache-dir \
    "torch_geometric>=2.6.0"

# Install PyG extensions (scatter, sparse, cluster)
# Build from source with --no-build-isolation for PyTorch compatibility
RUN pip install --no-cache-dir --no-build-isolation \
    torch_scatter \
    torch_sparse \
    torch_cluster

# ============================================
# Lab 4: MeshGraphNet
# DGL is no longer required; MeshGraphNet now uses PyTorch Geometric backend.
# psutil is still used for memory profiling.
# ============================================
RUN pip install --no-cache-dir \
    "psutil>=6.0.0"

# ============================================
# Lab 5: Diffusion / FWI
# ============================================
RUN pip install --no-cache-dir \
    "deepwave>=0.0.21"

# ============================================
# Jupyter Lab for interactive notebooks
# ============================================
RUN pip install --no-cache-dir \
    "jupyterlab>=4.0.0" \
    "ipywidgets>=8.0.0"

# ============================================
# Copy workshop materials
# ============================================
COPY . /workspace/physicsnemo_workshop/
WORKDIR /workspace/physicsnemo_workshop

# ============================================
# Download datasets (optional - can be done at runtime)
# ============================================
# Uncomment to pre-download datasets during build:
# RUN cd lab_2/raw_dataset && bash download_dataset.sh || true

# ============================================
# Expose Jupyter port
# ============================================
EXPOSE 8888

# ============================================
# Default command: Start Jupyter Lab
# ============================================
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--NotebookApp.token=''"]
