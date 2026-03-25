#!/usr/bin/env python3
"""
Script to divide video features into steps according to step annotations.
Creates a single npz file containing all processed videos with step-level features.
"""

import numpy as np
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple
import argparse


def load_annotations(annotation_path: str) -> Dict:
    """Load step annotations from JSON file."""
    with open(annotation_path, 'r') as f:
        annotations = json.load(f)
    return annotations


def load_video_features(feature_path: str) -> np.ndarray:
    """Load video features from npz file."""
    data = np.load(feature_path)
    # Assuming features are stored in 'arr_0' key
    features = data['arr_0']
    return features


def get_recording_id_from_filename(filename: str) -> str:
    """
    Extract recording_id from feature filename.
    Example: 9_2_360p_224.mp4_1s_1s.npz -> 9_2
    """
    base_name = os.path.basename(filename)
    # Remove the extension and suffix
    parts = base_name.replace('_360p_224.mp4_1s_1s.npz', '')
    return parts


def extract_step_features(features: np.ndarray, 
                          start_time: float, 
                          end_time: float, 
                          fps: float = 1.0) -> np.ndarray:
    """
    Extract features for a specific step based on time boundaries.
    
    Args:
        features: Video features array (num_frames, feature_dim)
        start_time: Step start time in seconds
        end_time: Step end time in seconds
        fps: Frames per second (default 1.0 for 1s_1s features)
    
    Returns:
        Features for the step
    """
    start_frame = int(start_time * fps)
    end_frame = int(end_time * fps)
    
    # Ensure we don't go out of bounds
    start_frame = max(0, start_frame)
    end_frame = min(len(features), end_frame)
    
    if start_frame >= end_frame:
        # Return empty array if invalid range
        return np.array([])
    
    return features[start_frame:end_frame]


def process_video(video_path: str, 
                  annotations: Dict, 
                  recording_id: str,
                  fps: float = 1.0) -> List[Dict]:
    """
    Process a single video, extracting features for each step.
    
    Returns:
        List of dictionaries containing step information and features
    """
    if recording_id not in annotations:
        print(f"Warning: No annotations found for recording_id: {recording_id}")
        return []
    
    # Load video features
    features = load_video_features(video_path)
    print(f"Processing {recording_id}: {features.shape[0]} frames, {features.shape[1]} features")
    
    video_annotations = annotations[recording_id]
    steps_data = []
    
    for step in video_annotations['steps']:
        step_features = extract_step_features(
            features, 
            step['start_time'], 
            step['end_time'], 
            fps
        )
        
        if len(step_features) > 0:
            step_info = {
                'recording_id': recording_id,
                'step_id': step['step_id'],
                'start_time': step['start_time'],
                'end_time': step['end_time'],
                'description': step['description'],
                'has_errors': step['has_errors'],
                'features': step_features,
                'num_frames': len(step_features)
            }
            steps_data.append(step_info)
    
    return steps_data


def save_all_steps_combined(all_steps: List[Dict], output_path: str):
    """
    Save all steps from all videos into a single npz file.
    Steps are grouped by recording_id.
    """
    save_dict = {}
    
    # Group steps by recording_id
    steps_by_video = {}
    for step_data in all_steps:
        recording_id = step_data['recording_id']
        if recording_id not in steps_by_video:
            steps_by_video[recording_id] = []
        steps_by_video[recording_id].append(step_data)
        
    for recording_id, steps in steps_by_video.items():
        # Store the number of steps for this recording
        save_dict[f"{recording_id}_num_steps"] = len(steps)
        
        for idx, step_data in enumerate(steps):
            prefix = f"{recording_id}_step_{idx:03d}"
            save_dict[f"{prefix}_step_id"] = step_data['step_id']
            save_dict[f"{prefix}_start_time"] = step_data['start_time']
            save_dict[f"{prefix}_end_time"] = step_data['end_time']
            save_dict[f"{prefix}_description"] = step_data['description']
            save_dict[f"{prefix}_has_errors"] = step_data['has_errors']
            save_dict[f"{prefix}_features"] = step_data['features']
            save_dict[f"{prefix}_num_frames"] = step_data['num_frames']
            
    # Save metadata
    save_dict['num_steps'] = len(all_steps)
    save_dict['recording_ids'] = list(steps_by_video.keys())
    
    np.savez_compressed(output_path, **save_dict)
    print(f"\nSaved {len(all_steps)} steps to {output_path}")


def save_steps_per_video(all_steps: List[Dict], output_dir: str):
    """
    Save steps for each video into separate npz files.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Group steps by recording_id
    steps_by_video = {}
    for step_data in all_steps:
        recording_id = step_data['recording_id']
        if recording_id not in steps_by_video:
            steps_by_video[recording_id] = []
        steps_by_video[recording_id].append(step_data)
    
    # Save each video's steps
    for recording_id, steps in steps_by_video.items():
        save_dict = {}
        
        for idx, step_data in enumerate(steps):
            prefix = f"step_{idx:03d}"
            save_dict[f"{prefix}_step_id"] = step_data['step_id']
            save_dict[f"{prefix}_start_time"] = step_data['start_time']
            save_dict[f"{prefix}_end_time"] = step_data['end_time']
            save_dict[f"{prefix}_description"] = step_data['description']
            save_dict[f"{prefix}_has_errors"] = step_data['has_errors']
            save_dict[f"{prefix}_features"] = step_data['features']
            save_dict[f"{prefix}_num_frames"] = step_data['num_frames']
        
        save_dict['recording_id'] = recording_id
        save_dict['num_steps'] = len(steps)
        
        output_path = os.path.join(output_dir, f"{recording_id}_steps.npz")
        np.savez_compressed(output_path, **save_dict)
        print(f"Saved {len(steps)} steps for {recording_id} to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Divide video features into steps according to annotations'
    )
    parser.add_argument(
        '--features_dir',
        type=str,
        default='video_features',
        help='Directory containing video feature files (.npz)'
    )
    parser.add_argument(
        '--annotations',
        type=str,
        default='annotations/annotation_json/step_annotations.json',
        help='Path to step annotations JSON file'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='video_features_by_steps.npz',
        help='Output file path for combined steps (single file mode)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='video_features_by_steps',
        help='Output directory for per-video files (separate files mode)'
    )
    parser.add_argument(
        '--mode',
        type=str,
        choices=['combined', 'separate'],
        default='combined',
        help='Output mode: "combined" for single file, "separate" for per-video files'
    )
    parser.add_argument(
        '--fps',
        type=float,
        default=1.0,
        help='Frames per second (1.0 for 1s_1s features)'
    )
    
    args = parser.parse_args()
    
    # Load annotations
    print(f"Loading annotations from {args.annotations}...")
    annotations = load_annotations(args.annotations)
    print(f"Loaded annotations for {len(annotations)} videos")
    
    # Get all video feature files
    feature_files = sorted(Path(args.features_dir).glob('*.npz'))
    print(f"Found {len(feature_files)} video feature files")
    
    # Process all videos
    all_steps = []
    for feature_file in feature_files:
        recording_id = get_recording_id_from_filename(str(feature_file))
        steps_data = process_video(str(feature_file), annotations, recording_id, args.fps)
        all_steps.extend(steps_data)
    
    print(f"\nTotal steps extracted: {len(all_steps)}")
    
    # Calculate statistics
    total_frames = sum(step['num_frames'] for step in all_steps)
    avg_frames_per_step = total_frames / len(all_steps) if all_steps else 0
    print(f"Total frames across all steps: {total_frames}")
    print(f"Average frames per step: {avg_frames_per_step:.2f}")
    
    # Save output
    if args.mode == 'combined':
        save_all_steps_combined(all_steps, args.output)
    else:
        save_steps_per_video(all_steps, args.output_dir)
    
    print("\nProcessing complete!")


if __name__ == '__main__':
    main()
