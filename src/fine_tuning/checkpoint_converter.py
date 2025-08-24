import h5py
import torch
import numpy as np
import os
from datetime import datetime

def convert_checkpoint_to_h5(checkpoint_path, h5_path):
    """
    Convert PyTorch checkpoint to HDF5 format.
    
    Args:
        checkpoint_path: Path to the PyTorch checkpoint (.pt file)
        h5_path: Path where to save the HDF5 file
    """
    print(f"Loading checkpoint from {checkpoint_path}")
    # Load PyTorch checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Handle different checkpoint formats
    if isinstance(checkpoint, dict):
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            # Assume the dict itself is the state dict
            state_dict = checkpoint
    else:
        # If checkpoint is not a dict, it might be the model itself
        state_dict = checkpoint.state_dict() if hasattr(checkpoint, 'state_dict') else checkpoint
    
    print(f"Creating HDF5 file at {h5_path}")
    # Create HDF5 file
    with h5py.File(h5_path, 'w') as f:
        # Save model state dict
        state_dict_group = f.create_group('state_dict')
        for key, value in state_dict.items():
            print(f"Converting layer: {key}")
            # Convert tensor to numpy and save
            if isinstance(value, torch.Tensor):
                state_dict_group.create_dataset(key, data=value.cpu().numpy())
            else:
                print(f"Warning: Skipping non-tensor parameter {key}")
        
        # Save model config if exists
        if isinstance(checkpoint, dict) and 'config' in checkpoint:
            config_group = f.create_group('config')
            for key, value in checkpoint['config'].items():
                if isinstance(value, (int, float, str)):
                    config_group.attrs[key] = value
                elif isinstance(value, dict):
                    config_subgroup = config_group.create_group(key)
                    for k, v in value.items():
                        if isinstance(v, (int, float, str)):
                            config_subgroup.attrs[k] = v
        
        # Save metadata
        f.attrs['format_version'] = '1.0'
        f.attrs['creation_date'] = str(datetime.now())
        f.attrs['original_file'] = os.path.basename(checkpoint_path)

def load_checkpoint_from_h5(h5_path):
    """
    Load model checkpoint from HDF5 format.
    
    Args:
        h5_path: Path to the HDF5 file
    
    Returns:
        dict: Checkpoint dictionary compatible with PyTorch models
    """
    print(f"Loading HDF5 file from {h5_path}")
    checkpoint = {'state_dict': {}, 'config': {}}
    
    with h5py.File(h5_path, 'r') as f:
        # Load state dict
        state_dict_group = f['state_dict']
        for key in state_dict_group.keys():
            print(f"Loading layer: {key}")
            # Convert numpy array to tensor
            checkpoint['state_dict'][key] = torch.from_numpy(state_dict_group[key][()])
        
        # Load config if exists
        if 'config' in f:
            config_group = f['config']
            # Load attributes
            for key in config_group.attrs:
                checkpoint['config'][key] = config_group.attrs[key]
            # Load subgroups
            for key in config_group.keys():
                checkpoint['config'][key] = {}
                for k in config_group[key].attrs:
                    checkpoint['config'][key][k] = config_group[key].attrs[k]
        
        # Load metadata
        if 'format_version' in f.attrs:
            checkpoint['format_version'] = f.attrs['format_version']
        if 'creation_date' in f.attrs:
            checkpoint['creation_date'] = f.attrs['creation_date']
    
    return checkpoint

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Convert between PyTorch checkpoint and HDF5')
    parser.add_argument('--input', required=True, help='Input checkpoint path')
    parser.add_argument('--output', required=True, help='Output file path')
    parser.add_argument('--to-h5', action='store_true', help='Convert to H5 (default: convert to PyTorch)')
    
    args = parser.parse_args()
    
    if args.to_h5:
        print(f"Converting {args.input} to HDF5 format...")
        convert_checkpoint_to_h5(args.input, args.output)
        
        # Print size comparison
        pt_size = os.path.getsize(args.input) / (1024 * 1024)
        h5_size = os.path.getsize(args.output) / (1024 * 1024)
        print(f"Original PT size: {pt_size:.2f}MB")
        print(f"HDF5 size: {h5_size:.2f}MB")
        print(f"Size reduction: {100 * (1 - h5_size/pt_size):.1f}%")
    else:
        print(f"Converting {args.input} to PyTorch format...")
        checkpoint = load_checkpoint_from_h5(args.input)
        torch.save(checkpoint, args.output)
        print("Conversion complete!")