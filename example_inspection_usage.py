#!/usr/bin/env python3
"""
Example script showing how to use the StepFeatureInspector class programmatically.
This demonstrates how to integrate the inspector into your own analysis pipeline.
"""

import sys
sys.path.append('.')
from inspect_step_features import StepFeatureInspector
import numpy as np


def example_basic_usage():
    """Basic usage example."""
    print("="*80)
    print("EXAMPLE 1: Basic Usage")
    print("="*80)
    
    # Initialize inspector
    inspector = StepFeatureInspector('video_features_by_steps.npz', mode='combined')
    
    # Get info for a specific step
    step_info = inspector.get_step_info(0)
    print(f"\nStep 0 info:")
    print(f"  Recording: {step_info['recording_id']}")
    print(f"  Description: {step_info['description']}")
    print(f"  Features shape: {step_info['features'].shape}")
    print(f"  Duration: {step_info['end_time'] - step_info['start_time']:.2f}s")


def example_batch_analysis():
    """Analyze multiple steps in batch."""
    print("\n" + "="*80)
    print("EXAMPLE 2: Batch Analysis")
    print("="*80)
    
    inspector = StepFeatureInspector('video_features_by_steps.npz', mode='combined')
    
    # Collect statistics for all steps
    step_lengths = []
    feature_means = []
    
    for i in range(inspector.num_steps):
        info = inspector.get_step_info(i)
        step_lengths.append(info['num_frames'])
        if len(info['features']) > 0:
            feature_means.append(np.mean(info['features']))
    
    print(f"\nAnalyzed {inspector.num_steps} steps:")
    print(f"  Average step length: {np.mean(step_lengths):.2f} frames")
    print(f"  Step length std: {np.std(step_lengths):.2f} frames")
    print(f"  Average feature mean: {np.mean(feature_means):.6f}")
    print(f"  Feature mean std: {np.std(feature_means):.6f}")


def example_custom_filtering():
    """Custom filtering example."""
    print("\n" + "="*80)
    print("EXAMPLE 3: Custom Filtering")
    print("="*80)
    
    inspector = StepFeatureInspector('video_features_by_steps.npz', mode='combined')
    
    # Find all short steps (< 20 seconds) without errors
    short_clean_steps = []
    
    for i in range(inspector.num_steps):
        info = inspector.get_step_info(i)
        duration = info['end_time'] - info['start_time']
        
        if duration < 20 and not info['has_errors']:
            short_clean_steps.append((i, info))
    
    print(f"\nFound {len(short_clean_steps)} short steps without errors:")
    for idx, info in short_clean_steps[:5]:  # Show first 5
        duration = info['end_time'] - info['start_time']
        print(f"  Step {idx}: {info['description'][:50]} ({duration:.2f}s)")


def example_feature_extraction():
    """Extract features for machine learning."""
    print("\n" + "="*80)
    print("EXAMPLE 4: Feature Extraction for ML")
    print("="*80)
    
    inspector = StepFeatureInspector('video_features_by_steps.npz', mode='combined')
    
    # Extract features and labels for a subset of steps
    X_list = []
    y_list = []
    
    for i in range(min(10, inspector.num_steps)):  # Process first 10 steps
        info = inspector.get_step_info(i)
        
        # Use features (you might want to aggregate them)
        features = info['features']
        if len(features) > 0:
            # Example: use mean of features across frames
            X_list.append(np.mean(features, axis=0))
            # Example label: step_id
            y_list.append(info['step_id'])
    
    X = np.array(X_list)
    y = np.array(y_list)
    
    print(f"\nExtracted features for ML:")
    print(f"  X shape: {X.shape}")
    print(f"  y shape: {y.shape}")
    print(f"  Unique step IDs: {len(np.unique(y))}")


def example_temporal_analysis():
    """Analyze temporal patterns."""
    print("\n" + "="*80)
    print("EXAMPLE 5: Temporal Pattern Analysis")
    print("="*80)
    
    inspector = StepFeatureInspector('video_features_by_steps.npz', mode='combined')
    
    # Group steps by recording and analyze temporal order
    from collections import defaultdict
    recording_steps = defaultdict(list)
    
    for i in range(inspector.num_steps):
        info = inspector.get_step_info(i)
        recording_steps[info['recording_id']].append((info['start_time'], i, info))
    
    # Analyze one recording
    sample_recording = list(recording_steps.keys())[0]
    steps = sorted(recording_steps[sample_recording])
    
    print(f"\nTemporal analysis for recording '{sample_recording}':")
    print(f"  Number of steps: {len(steps)}")
    
    for start_time, idx, info in steps[:5]:  # Show first 5
        duration = info['end_time'] - info['start_time']
        print(f"  {start_time:7.2f}s: {info['description'][:45]:45s} ({duration:5.2f}s)")


def main():
    """Run all examples."""
    try:
        example_basic_usage()
        example_batch_analysis()
        example_custom_filtering()
        example_feature_extraction()
        example_temporal_analysis()
        
        print("\n" + "="*80)
        print("All examples completed successfully!")
        print("="*80)
        
    except FileNotFoundError:
        print("Error: video_features_by_steps.npz not found.")
        print("Please run divide_features_by_steps.py first.")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
