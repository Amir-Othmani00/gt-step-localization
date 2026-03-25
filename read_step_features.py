#!/usr/bin/env python3
"""
Helper script to read and explore the processed step features.
"""

import numpy as np
import argparse


def read_combined_file(file_path: str):
    """Read and display information from a combined steps file."""
    data = np.load(file_path, allow_pickle=True)
    
    num_steps = int(data['num_steps'])
    recording_ids = data['recording_ids']
    print(f"=== Combined Step Features File ===")
    print(f"Total number of steps: {num_steps}")
    print(f"Total recordings: {len(recording_ids)}")
    print(f"\nFirst recording details:")
    print("-" * 80)
    
    # Show first recording info
    if len(recording_ids) > 0:
        recording_id = recording_ids[0]
        rec_num_steps = int(data[f"{recording_id}_num_steps"])
        print(f"\nRecording {recording_id}:")
        print(f"  Number of steps: {rec_num_steps}")
        for idx in range(min(5, rec_num_steps)):
            prefix = f"{recording_id}_step_{idx:03d}"
            step_id = int(data[f"{prefix}_step_id"])
            start_time = float(data[f"{prefix}_start_time"])
            end_time = float(data[f"{prefix}_end_time"])
            description = str(data[f"{prefix}_description"])
            num_frames = int(data[f"{prefix}_num_frames"])
            features_shape = data[f"{prefix}_features"].shape
            
            print(f"\n  Step {idx}:")
            print(f"    Step ID: {step_id}")
            print(f"    Time: {start_time:.2f}s - {end_time:.2f}s ({end_time-start_time:.2f}s)")
            print(f"    Description: {description}")
            print(f"    Frames: {num_frames}")
            print(f"    Features shape: {features_shape}")
    
    # Calculate statistics
    total_frames = 0
    feature_dims = set()
    
    for recording_id in recording_ids:
        rec_num_steps = int(data[f"{recording_id}_num_steps"])
        for idx in range(rec_num_steps):
            prefix = f"{recording_id}_step_{idx:03d}"
            num_frames = int(data[f"{prefix}_num_frames"])
            features = data[f"{prefix}_features"]
            total_frames += num_frames
            if len(features.shape) > 1:
                feature_dims.add(features.shape[1])
    
    print("\n" + "=" * 80)
    print("Summary Statistics:")
    print(f"  Total frames: {total_frames}")
    if num_steps > 0:
        print(f"  Average frames per step: {total_frames/num_steps:.2f}")
    print(f"  Feature dimensions: {feature_dims}")


def read_separate_file(file_path: str):
    """Read and display information from a per-video steps file."""
    data = np.load(file_path, allow_pickle=True)
    
    recording_id = str(data['recording_id'])
    num_steps = int(data['num_steps'])
    
    print(f"=== Video: {recording_id} ===")
    print(f"Number of steps: {num_steps}")
    print("\nSteps:")
    print("-" * 80)
    
    for idx in range(num_steps):
        prefix = f"step_{idx:03d}"
        step_id = int(data[f"{prefix}_step_id"])
        start_time = float(data[f"{prefix}_start_time"])
        end_time = float(data[f"{prefix}_end_time"])
        description = str(data[f"{prefix}_description"])
        num_frames = int(data[f"{prefix}_num_frames"])
        features_shape = data[f"{prefix}_features"].shape
        has_errors = bool(data[f"{prefix}_has_errors"])
        
        print(f"\nStep {idx} (ID: {step_id}):")
        print(f"  Time: {start_time:.2f}s - {end_time:.2f}s ({end_time-start_time:.2f}s)")
        print(f"  Description: {description}")
        print(f"  Frames: {num_frames}")
        print(f"  Features shape: {features_shape}")
        print(f"  Has errors: {has_errors}")


def main():
    parser = argparse.ArgumentParser(
        description='Read and explore processed step features'
    )
    parser.add_argument(
        'file_path',
        type=str,
        help='Path to the processed features file (.npz)'
    )
    parser.add_argument(
        '--mode',
        type=str,
        choices=['combined', 'separate'],
        default='combined',
        help='File mode: "combined" or "separate"'
    )
    
    args = parser.parse_args()
    
    if args.mode == 'combined':
        read_combined_file(args.file_path)
    else:
        read_separate_file(args.file_path)


if __name__ == '__main__':
    main()
