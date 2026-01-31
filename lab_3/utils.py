# SPDX-FileCopyrightText: Copyright (c) 2023 - 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Utility functions for Lab 5: Transolver"""

import os
import subprocess
import zipfile
import shutil
import numpy as np

# Try to import pyvista for VTP files
try:
    import pyvista as pv
    HAS_PYVISTA = True
except ImportError:
    HAS_PYVISTA = False


def softmax(x, axis=-1):
    """Numerically stable softmax."""
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / e_x.sum(axis=axis, keepdims=True)


def download_stokes_dataset(data_dir="./dataset", raw_dir="./results", max_samples=100):
    """
    Download Stokes flow dataset from NGC.
    
    Args:
        data_dir: Directory to store processed VTP files
        raw_dir: Temporary directory for raw download
        max_samples: Maximum number of samples to keep
    
    Returns:
        bool: True if dataset is available, False otherwise
    """
    if os.path.exists(data_dir) and len(os.listdir(data_dir)) > 0:
        print(f"✓ Dataset already exists at {data_dir} ({len(os.listdir(data_dir))} files)")
        return True
    
    print("Downloading Stokes flow dataset from NGC...")
    url = 'https://api.ngc.nvidia.com/v2/resources/org/nvidia/team/physicsnemo/physicsnemo_datasets_stokes_flow/0.0.1/files?redirect=true&path=results_polygon.zip'
    
    try:
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
        
        # Move VTP files
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
        
    except Exception as e:
        print(f"Download failed: {e}")
        print("Will use synthetic data instead.")
        return False


def load_stokes_sample(data_dir="./dataset", sample_idx=0):
    """
    Load a sample VTP file from the Stokes dataset.
    
    Args:
        data_dir: Directory containing VTP files
        sample_idx: Index of sample to load
    
    Returns:
        tuple: (coords, u, v, p) arrays
    """
    if HAS_PYVISTA and os.path.exists(data_dir):
        vtp_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.vtp')])
        if vtp_files and sample_idx < len(vtp_files):
            mesh = pv.read(os.path.join(data_dir, vtp_files[sample_idx]))
            coords = np.array(mesh.points[:, :2])  # 2D coordinates
            u = np.array(mesh.point_data.get('u', np.zeros(len(coords))))
            v = np.array(mesh.point_data.get('v', np.zeros(len(coords))))
            p = np.array(mesh.point_data.get('p', np.zeros(len(coords))))
            print(f"✓ Loaded {vtp_files[sample_idx]} ({len(coords)} points)")
            return coords, u, v, p
    
    # Fallback: Create synthetic Stokes-like data
    print("Using synthetic data (dataset not found or pyvista not installed)")
    return create_synthetic_stokes_data()


def create_synthetic_stokes_data(nx=60, ny=25):
    """
    Create synthetic Stokes flow data for testing.
    
    Args:
        nx, ny: Grid dimensions
    
    Returns:
        tuple: (coords, u, v, p) arrays
    """
    x = np.linspace(0, 2.2, nx)
    y = np.linspace(0, 0.4, ny)
    X, Y = np.meshgrid(x, y)
    coords = np.stack([X.flatten(), Y.flatten()], axis=1)
    
    # Parabolic velocity profile
    u = 0.3 * 4 * coords[:, 1] * (0.4 - coords[:, 1]) / 0.4**2
    v = np.zeros(len(coords))
    p = -coords[:, 0] * 0.1  # Linear pressure drop
    
    print(f"✓ Created synthetic data ({len(coords)} points)")
    return coords, u, v, p


def get_num_samples(data_dir="./dataset"):
    """Get number of available samples in dataset."""
    if os.path.exists(data_dir):
        return len([f for f in os.listdir(data_dir) if f.endswith('.vtp')])
    return 0
