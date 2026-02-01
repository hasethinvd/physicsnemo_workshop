#!/bin/bash
# PhysicsNeMo Workshop - Docker Build Script

set -e

# Configuration
IMAGE_NAME="physicsnemo-workshop"
IMAGE_TAG="${IMAGE_TAG:-latest}"
BASE_CONTAINER="${BASE_CONTAINER:-nvcr.io/nvidia/pytorch:24.12-py3}"
PHYSICSNEMO_BRANCH="${PHYSICSNEMO_BRANCH:-main}"
PHYSICSNEMO_SYM_BRANCH="${PHYSICSNEMO_SYM_BRANCH:-main}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}================================================${NC}"
echo -e "${GREEN}  PhysicsNeMo Workshop Docker Build${NC}"
echo -e "${GREEN}================================================${NC}"
echo ""
echo -e "Base container:        ${YELLOW}${BASE_CONTAINER}${NC}"
echo -e "PhysicsNeMo branch:    ${YELLOW}${PHYSICSNEMO_BRANCH}${NC}"
echo -e "PhysicsNeMo-Sym branch: ${YELLOW}${PHYSICSNEMO_SYM_BRANCH}${NC}"
echo -e "Image name:            ${YELLOW}${IMAGE_NAME}:${IMAGE_TAG}${NC}"
echo ""

# Navigate to repository root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/.."

# Build the Docker image
echo -e "${GREEN}Building Docker image...${NC}"
docker build \
    --build-arg BASE_CONTAINER="${BASE_CONTAINER}" \
    --build-arg PHYSICSNEMO_BRANCH="${PHYSICSNEMO_BRANCH}" \
    --build-arg PHYSICSNEMO_SYM_BRANCH="${PHYSICSNEMO_SYM_BRANCH}" \
    -t "${IMAGE_NAME}:${IMAGE_TAG}" \
    -f Dockerfile \
    .

echo ""
echo -e "${GREEN}================================================${NC}"
echo -e "${GREEN}  Build Complete!${NC}"
echo -e "${GREEN}================================================${NC}"
echo ""
echo "Run the container with:"
echo ""
echo -e "  ${YELLOW}docker run --gpus all -it \\${NC}"
echo -e "  ${YELLOW}  -p 8888:8888 \\${NC}"
echo -e "  ${YELLOW}  -v \$(pwd):/workspace/physicsnemo_workshop \\${NC}"
echo -e "  ${YELLOW}  --shm-size=16gb \\${NC}"
echo -e "  ${YELLOW}  ${IMAGE_NAME}:${IMAGE_TAG}${NC}"
echo ""
echo "Or use docker compose:"
echo ""
echo -e "  ${YELLOW}docker compose up${NC}"
echo ""
echo "To build with a specific branch:"
echo ""
echo -e "  ${YELLOW}PHYSICSNEMO_BRANCH=v2.0.0 PHYSICSNEMO_SYM_BRANCH=v2.0.0 ./scripts/build_docker.sh${NC}"
echo ""
