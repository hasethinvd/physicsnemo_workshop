#!/bin/bash
# PhysicsNeMo Workshop - Environment Setup Script
# Works with pip or uv
#
# Usage:
#   ./scripts/setup_env.sh          # Use pip
#   ./scripts/setup_env.sh --uv     # Use uv (faster)

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Check for uv flag
USE_UV=false
if [[ "$1" == "--uv" ]]; then
    USE_UV=true
fi

# Detect package manager
if $USE_UV; then
    if ! command -v uv &> /dev/null; then
        echo -e "${YELLOW}Installing uv...${NC}"
        curl -LsSf https://astral.sh/uv/install.sh | sh
        source $HOME/.cargo/env
    fi
    PIP="uv pip"
    echo -e "${GREEN}Using uv for package management${NC}"
else
    PIP="pip"
    echo -e "${GREEN}Using pip for package management${NC}"
fi

echo ""
echo -e "${GREEN}================================================${NC}"
echo -e "${GREEN}  PhysicsNeMo Workshop Environment Setup${NC}"
echo -e "${GREEN}================================================${NC}"
echo ""

# Check Python version
PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo -e "Python version: ${YELLOW}${PYTHON_VERSION}${NC}"

# Check CUDA version from nvidia-smi or nvcc
if command -v nvidia-smi &> /dev/null; then
    CUDA_VERSION=$(nvidia-smi | grep "CUDA Version" | sed 's/.*CUDA Version: //' | sed 's/ .*//' | cut -d'.' -f1,2)
    echo -e "CUDA version (nvidia-smi): ${YELLOW}${CUDA_VERSION}${NC}"
elif command -v nvcc &> /dev/null; then
    CUDA_VERSION=$(nvcc --version | grep "release" | sed 's/.*release //' | sed 's/,.*//')
    echo -e "CUDA version (nvcc): ${YELLOW}${CUDA_VERSION}${NC}"
else
    echo -e "${YELLOW}CUDA not found - will install CPU-only versions${NC}"
    CUDA_VERSION="cpu"
fi

# Map CUDA version to wheel suffix
case "$CUDA_VERSION" in
    12.4*|12.5*|12.6*) CUDA_WHEEL="cu124" ;;
    12.1*|12.2*|12.3*) CUDA_WHEEL="cu121" ;;
    11.8*) CUDA_WHEEL="cu118" ;;
    11.7*) CUDA_WHEEL="cu117" ;;
    cpu) CUDA_WHEEL="cpu" ;;
    *) CUDA_WHEEL="cu121" ; echo -e "${YELLOW}Unknown CUDA $CUDA_VERSION, defaulting to cu121${NC}" ;;
esac
echo -e "Using wheel suffix: ${YELLOW}${CUDA_WHEEL}${NC}"

echo ""

# Step 1: Core dependencies
echo -e "${GREEN}[1/6] Installing core dependencies...${NC}"
$PIP install --upgrade pip setuptools wheel
$PIP install "numpy>=1.24.0,<2.0" "scipy>=1.10.0" "h5py>=3.7.0" "matplotlib>=3.8.0" "einops>=0.7.0"

# Step 2: Config & tracking
echo -e "${GREEN}[2/6] Installing config and tracking tools...${NC}"
$PIP install "hydra-core>=1.3.0" "omegaconf>=2.3.0" "wandb>=0.15.1" "mlflow>=2.1.1" "termcolor>=2.1.1" "tqdm>=4.60.0"

# Step 3: Visualization
echo -e "${GREEN}[3/6] Installing visualization tools...${NC}"
$PIP install "vtk>=9.2.6" "pyvista>=0.43.0" "plotly>=5.18.0" "imageio>=2.31.0"

# Step 4: Jupyter
echo -e "${GREEN}[4/6] Installing Jupyter...${NC}"
$PIP install "jupyterlab>=4.0.0" "ipywidgets>=8.0.0"

# Step 5: PyTorch Geometric (Lab 3: xMGN)
echo -e "${GREEN}[5/6] Installing PyTorch Geometric...${NC}"
$PIP install "torch_geometric>=2.6.0"
# Note: torch_scatter, torch_sparse, torch_cluster may need manual install
# based on your PyTorch/CUDA version

# Step 6: DGL (Lab 4: MeshGraphNet)
echo -e "${GREEN}[6/6] Installing DGL...${NC}"
if [[ "$CUDA_WHEEL" != "cpu" ]]; then
    # Try to install DGL for detected CUDA version
    $PIP install dgl -f https://data.dgl.ai/wheels/torch-2.4/${CUDA_WHEEL}/repo.html || \
    $PIP install dgl -f https://data.dgl.ai/wheels/torch-2.3/${CUDA_WHEEL}/repo.html || \
    echo -e "${YELLOW}DGL installation failed - install manually${NC}"
else
    $PIP install dgl
fi

# Step 7: PyG extensions (for Lab 3: xMGN)
echo -e "${GREEN}[7/7] Installing PyG extensions...${NC}"
if [[ "$CUDA_WHEEL" != "cpu" ]]; then
    $PIP install torch_scatter torch_sparse torch_cluster \
        -f https://data.pyg.org/whl/torch-2.4.0+${CUDA_WHEEL}.html || \
    echo -e "${YELLOW}PyG extensions failed - may need manual install${NC}"
fi

echo ""
echo -e "${GREEN}================================================${NC}"
echo -e "${GREEN}  PhysicsNeMo Installation${NC}"
echo -e "${GREEN}================================================${NC}"
echo ""

# Clone and install PhysicsNeMo
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSHOP_DIR="$(dirname "$SCRIPT_DIR")"

if [ ! -d "$WORKSHOP_DIR/physicsnemo" ]; then
    echo -e "${GREEN}Cloning PhysicsNeMo...${NC}"
    git clone --depth 1 https://github.com/NVIDIA/physicsnemo.git "$WORKSHOP_DIR/physicsnemo"
fi
echo -e "${GREEN}Installing PhysicsNeMo...${NC}"
$PIP install -e "$WORKSHOP_DIR/physicsnemo"

# Clone and install PhysicsNeMo-Sym
if [ ! -d "$WORKSHOP_DIR/physicsnemo-sym" ]; then
    echo -e "${GREEN}Cloning PhysicsNeMo-Sym...${NC}"
    git clone --depth 1 https://github.com/NVIDIA/physicsnemo-sym.git "$WORKSHOP_DIR/physicsnemo-sym"
fi
echo -e "${GREEN}Installing PhysicsNeMo-Sym...${NC}"

# Set CUDA architectures based on CUDA version to avoid compilation errors
# compute_100 (Blackwell) requires CUDA 12.8+, compute_90 (Hopper) requires CUDA 12.0+
case "$CUDA_VERSION" in
    12.8*|12.9*|12.10*|13.*)
        # CUDA 12.8+ supports Blackwell (compute_100)
        export TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;9.0;10.0"
        ;;
    12.0*|12.1*|12.2*|12.3*|12.4*|12.5*|12.6*|12.7*)
        # CUDA 12.0-12.7 supports up to Hopper (compute_90)
        export TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;9.0"
        ;;
    11.8*)
        # CUDA 11.8 supports up to Hopper (compute_90)
        export TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;9.0"
        ;;
    *)
        # Default for older CUDA versions
        export TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6"
        ;;
esac
echo -e "Building for CUDA architectures: ${YELLOW}${TORCH_CUDA_ARCH_LIST}${NC}"

$PIP install --no-build-isolation -e "$WORKSHOP_DIR/physicsnemo-sym"

echo ""
echo -e "${GREEN}================================================${NC}"
echo -e "${GREEN}  Setup Complete!${NC}"
echo -e "${GREEN}================================================${NC}"
echo ""
echo "To verify installation:"
echo "  python -c \"import physicsnemo; print('PhysicsNeMo:', physicsnemo.__version__)\""
echo ""
echo "To start Jupyter Lab:"
echo "  jupyter lab"
echo ""
echo -e "${YELLOW}Note: For PyG extensions (torch_scatter, torch_sparse, torch_cluster),${NC}"
echo -e "${YELLOW}you may need to install manually based on your PyTorch/CUDA version:${NC}"
echo "  pip install torch_scatter torch_sparse torch_cluster -f https://data.pyg.org/whl/torch-2.x.0+cuXXX.html"
echo ""
