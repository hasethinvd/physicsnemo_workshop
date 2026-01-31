# SPDX-FileCopyrightText: Copyright (c) 2023 - 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Utility functions for Lab 3: Transolver"""

import os
import subprocess
import zipfile
import shutil
import numpy as np

# Require pyvista for VTP files
import pyvista as pv


def softmax(x, axis=-1):
    """Numerically stable softmax."""
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / e_x.sum(axis=axis, keepdims=True)


def download_stokes_dataset(data_dir="../lab_2/dataset", max_samples=100):
    """
    Download Stokes flow dataset from NGC (uses same location as Lab 2).
    
    Args:
        data_dir: Directory to store processed VTP files (default: ../lab_2/dataset)
        max_samples: Maximum number of samples to keep
    
    Returns:
        bool: True if dataset is available
    """
    # Check if dataset already exists
    if os.path.exists(data_dir):
        vtp_files = [f for f in os.listdir(data_dir) if f.endswith('.vtp')]
        if vtp_files:
            print(f"✓ Dataset already exists at {data_dir} ({len(vtp_files)} VTP files)")
            return True
    
    print("Downloading Stokes flow dataset from NGC...")
    url = 'https://api.ngc.nvidia.com/v2/resources/org/nvidia/team/physicsnemo/physicsnemo_datasets_stokes_flow/0.0.1/files?redirect=true&path=results_polygon.zip'
    
    # Download
    subprocess.run(
        ['wget', '--content-disposition', url, '-O', 'results_polygon.zip'], 
        check=True
    )
    
    # Extract
    with zipfile.ZipFile('results_polygon.zip', 'r') as zip_ref:
        zip_ref.extractall('.')
    
    # Create dataset directory
    os.makedirs(data_dir, exist_ok=True)
    
    # Move VTP files from extracted 'results' folder
    raw_dir = './results'
    if os.path.exists(raw_dir):
        vtp_files = sorted([f for f in os.listdir(raw_dir) if f.endswith('.vtp')])
        for f in vtp_files[:max_samples]:
            shutil.copy(os.path.join(raw_dir, f), data_dir)
        # Cleanup raw dir
        shutil.rmtree(raw_dir, ignore_errors=True)
    
    # Cleanup zip
    if os.path.exists('results_polygon.zip'):
        os.remove('results_polygon.zip')
        
    print(f"✓ Downloaded {len(os.listdir(data_dir))} samples to {data_dir}")
    return True


def load_stokes_sample(data_dir="../lab_2/dataset", sample_idx=0):
    """
    Load a sample VTP file from the Stokes dataset.
    
    Args:
        data_dir: Directory containing VTP files
        sample_idx: Index of sample to load
    
    Returns:
        tuple: (coords, u, v, p) arrays
    """
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Dataset not found at {data_dir}. Run download_stokes_dataset() first.")
    
    vtp_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.vtp')])
    if not vtp_files:
        raise FileNotFoundError(f"No VTP files found in {data_dir}. Run download_stokes_dataset() first.")
    
    if sample_idx >= len(vtp_files):
        raise IndexError(f"Sample index {sample_idx} out of range (only {len(vtp_files)} files)")
    
    mesh = pv.read(os.path.join(data_dir, vtp_files[sample_idx]))
    coords = np.array(mesh.points[:, :2])  # 2D coordinates
    u = np.array(mesh.point_data.get('u', np.zeros(len(coords))))
    v = np.array(mesh.point_data.get('v', np.zeros(len(coords))))
    p = np.array(mesh.point_data.get('p', np.zeros(len(coords))))
    print(f"✓ Loaded {vtp_files[sample_idx]} ({len(coords)} points)")
    return coords, u, v, p


def get_num_samples(data_dir="../lab_2/dataset"):
    """Get number of available samples in dataset."""
    if os.path.exists(data_dir):
        return len([f for f in os.listdir(data_dir) if f.endswith('.vtp')])
    return 0
