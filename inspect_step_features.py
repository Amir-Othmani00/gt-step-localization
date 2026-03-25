#!/usr/bin/env python3
"""
Comprehensive script to inspect and analyze video features per step.
Provides various analysis tools for exploring step-level features.
"""

import numpy as np
import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional


class StepFeatureInspector:
    """Inspector class for step-level features."""
    
    def __init__(self, file_path: str, mode: str = 'combined'):
        """
        Initialize the inspector.
        
        Args:
            file_path: Path to the .npz file
            mode: 'combined' or 'separate'
        """
        self.file_path = file_path
        self.mode = mode
        self.data = np.load(file_path, allow_pickle=True)
        
        if mode == 'combined':
            self.num_steps = int(self.data['num_steps'])
            self.recording_id = None
            self.recording_ids = [str(r) for r in self.data['recording_ids']]
            
            # Map flat step index to (recording_id, local_index)
            self.index_map = []
            for rid in self.recording_ids:
                rec_num_steps = int(self.data[f"{rid}_num_steps"])
                for local_idx in range(rec_num_steps):
                    self.index_map.append((rid, local_idx))
        else:
            self.recording_id = str(self.data['recording_id'])
            self.num_steps = int(self.data['num_steps'])
    
    def get_step_info(self, step_idx: int) -> Dict:
        """Get information for a specific step."""
        if step_idx < 0 or step_idx >= self.num_steps:
            raise ValueError(f"Step index {step_idx} out of range [0, {self.num_steps})")
        
        if self.mode == 'combined':
            rid, local_idx = self.index_map[step_idx]
            prefix = f"{rid}_step_{local_idx:03d}"
            return {
                'recording_id': rid,
                'step_id': int(self.data[f"{prefix}_step_id"]),
                'start_time': float(self.data[f"{prefix}_start_time"]),
                'end_time': float(self.data[f"{prefix}_end_time"]),
                'description': str(self.data[f"{prefix}_description"]),
                'has_errors': bool(self.data[f"{prefix}_has_errors"]),
                'features': self.data[f"{prefix}_features"],
                'num_frames': int(self.data[f"{prefix}_num_frames"])
            }
        else:
            prefix = f"step_{step_idx:03d}"
            return {
                'recording_id': self.recording_id,
                'step_id': int(self.data[f"{prefix}_step_id"]),
                'start_time': float(self.data[f"{prefix}_start_time"]),
                'end_time': float(self.data[f"{prefix}_end_time"]),
                'description': str(self.data[f"{prefix}_description"]),
                'has_errors': bool(self.data[f"{prefix}_has_errors"]),
                'features': self.data[f"{prefix}_features"],
                'num_frames': int(self.data[f"{prefix}_num_frames"])
            }
    
    def show_step(self, step_idx: int, show_stats: bool = True):
        """Display detailed information about a step."""
        info = self.get_step_info(step_idx)
        features = info['features']
        
        print(f"\n{'='*80}")
        print(f"STEP {step_idx}")
        print(f"{'='*80}")
        print(f"Recording ID:    {info['recording_id']}")
        print(f"Step ID:         {info['step_id']}")
        print(f"Time Range:      {info['start_time']:.2f}s - {info['end_time']:.2f}s "
              f"(duration: {info['end_time'] - info['start_time']:.2f}s)")
        print(f"Description:     {info['description']}")
        print(f"Has Errors:      {info['has_errors']}")
        print(f"Number of Frames: {info['num_frames']}")
        print(f"Features Shape:  {features.shape}")
        
        if show_stats and len(features) > 0:
            print(f"\nFeature Statistics:")
            print(f"  Mean:          {np.mean(features):.4f}")
            print(f"  Std:           {np.std(features):.4f}")
            print(f"  Min:           {np.min(features):.4f}")
            print(f"  Max:           {np.max(features):.4f}")
            print(f"  25th percentile: {np.percentile(features, 25):.4f}")
            print(f"  Median:        {np.median(features):.4f}")
            print(f"  75th percentile: {np.percentile(features, 75):.4f}")
            
            # Per-frame statistics
            frame_means = np.mean(features, axis=1)
            print(f"\nPer-Frame Mean Statistics:")
            print(f"  Min frame mean:  {np.min(frame_means):.4f}")
            print(f"  Max frame mean:  {np.max(frame_means):.4f}")
            print(f"  Std of means:    {np.std(frame_means):.4f}")
    
    def list_all_steps(self, show_details: bool = False):
        """List all steps in the file, grouped by recording."""
        if self.mode == 'separate':
            print(f"\n{'='*80}")
            print(f"ALL STEPS FOR VIDEO: {self.recording_id}")
            print(f"Total Steps: {self.num_steps}")
            print(f"{'='*80}\n")
            for idx in range(self.num_steps):
                info = self.get_step_info(idx)
                self._print_step_line(idx, info, show_details)
            return

        # Combined mode: group by recording_id
        # Collect steps per recording in order
        from collections import OrderedDict
        recordings: OrderedDict = OrderedDict()
        for idx in range(self.num_steps):
            info = self.get_step_info(idx)
            rid = info['recording_id']
            if rid not in recordings:
                recordings[rid] = []
            recordings[rid].append((idx, info))

        print(f"\nTotal steps: {self.num_steps} across {len(recordings)} recordings\n")
        for rid, steps in recordings.items():
            print(f"\n{'='*80}")
            print(f"VIDEO: {rid}  ({len(steps)} steps)")
            print(f"{'='*80}")
            for local_idx, (global_idx, info) in enumerate(steps):
                self._print_step_line(local_idx, info, show_details)

    def _print_step_line(self, idx: int, info: dict, show_details: bool):
        """Print a single step line (compact or detailed)."""
        duration = info['end_time'] - info['start_time']
        if show_details:
            print(f"Step {idx:3d} | ID: {info['step_id']:3d} | "
                  f"Time: {info['start_time']:7.2f}s-{info['end_time']:7.2f}s "
                  f"({duration:6.2f}s) | "
                  f"Frames: {info['num_frames']:3d} | "
                  f"Shape: {str(info['features'].shape):15s}")
            print(f"         Description: {info['description']}")
            if info['has_errors']:
                print(f"         ⚠️  Has errors!")
            print()
        else:
            error_flag = "⚠️ " if info['has_errors'] else ""
            print(f"  {idx:3d}. {error_flag}{info['description'][:60]:60s} | "
                  f"{info['num_frames']:3d} frames | "
                  f"{duration:6.2f}s")
    
    def search_steps(self, query: str, field: str = 'description'):
        """Search for steps matching a query."""
        results = []
        
        for idx in range(self.num_steps):
            info = self.get_step_info(idx)
            
            if field == 'description' and query.lower() in info['description'].lower():
                results.append((idx, info))
            elif field == 'recording_id' and query in info['recording_id']:
                results.append((idx, info))
            elif field == 'step_id' and info['step_id'] == int(query):
                results.append((idx, info))
        
        print(f"\nFound {len(results)} steps matching '{query}' in {field}:")
        print(f"{'='*80}\n")
        
        for idx, info in results:
            duration = info['end_time'] - info['start_time']
            print(f"Step {idx:3d} | Recording: {info['recording_id']:10s} | "
                  f"Frames: {info['num_frames']:3d} | Duration: {duration:6.2f}s")
            print(f"         {info['description']}")
            print()
        
        return results
    
    def compare_steps(self, step_indices: List[int]):
        """Compare multiple steps side by side."""
        print(f"\n{'='*80}")
        print(f"COMPARING {len(step_indices)} STEPS")
        print(f"{'='*80}\n")
        
        for idx in step_indices:
            info = self.get_step_info(idx)
            features = info['features']
            duration = info['end_time'] - info['start_time']
            
            print(f"Step {idx}:")
            print(f"  Recording:   {info['recording_id']}")
            print(f"  Description: {info['description'][:60]}")
            print(f"  Duration:    {duration:.2f}s")
            print(f"  Frames:      {info['num_frames']}")
            print(f"  Shape:       {features.shape}")
            if len(features) > 0:
                print(f"  Mean:        {np.mean(features):.4f}")
                print(f"  Std:         {np.std(features):.4f}")
            print()
    
    def get_statistics_summary(self):
        """Get overall statistics about all steps."""
        all_durations = []
        all_num_frames = []
        all_feature_means = []
        recordings = set()
        step_ids = set()
        errors_count = 0
        steps_per_video = {}
        
        for idx in range(self.num_steps):
            info = self.get_step_info(idx)
            duration = info['end_time'] - info['start_time']
            all_durations.append(duration)
            all_num_frames.append(info['num_frames'])
            recordings.add(info['recording_id'])
            step_ids.add(info['step_id'])
            steps_per_video[info['recording_id']] = steps_per_video.get(info['recording_id'], 0) + 1
            
            if info['has_errors']:
                errors_count += 1
            
            if len(info['features']) > 0:
                all_feature_means.append(np.mean(info['features']))
        
        print(f"\n{'='*80}")
        print(f"OVERALL STATISTICS")
        print(f"{'='*80}")
        print(f"\nDataset Overview:")
        print(f"  Total steps:       {self.num_steps}")
        print(f"  Unique recordings: {len(recordings)}")
        print(f"  Unique step IDs:   {len(step_ids)}")
        print(f"  Steps with errors: {errors_count} ({100*errors_count/self.num_steps:.1f}%)")
        
        spv = list(steps_per_video.values())
        print(f"\nSteps per Video:")
        print(f"  Mean:     {np.mean(spv):.2f}")
        print(f"  Std:      {np.std(spv):.2f}")
        print(f"  Min:      {np.min(spv)}")
        print(f"  Max:      {np.max(spv)}")
        print(f"  Median:   {np.median(spv):.0f}")

        print(f"\nDuration Statistics (seconds):")
        print(f"  Mean:     {np.mean(all_durations):.2f}")
        print(f"  Std:      {np.std(all_durations):.2f}")
        print(f"  Min:      {np.min(all_durations):.2f}")
        print(f"  Max:      {np.max(all_durations):.2f}")
        print(f"  Median:   {np.median(all_durations):.2f}")
        
        print(f"\nFrames per Step:")
        print(f"  Mean:     {np.mean(all_num_frames):.2f}")
        print(f"  Std:      {np.std(all_num_frames):.2f}")
        print(f"  Min:      {np.min(all_num_frames)}")
        print(f"  Max:      {np.max(all_num_frames)}")
        print(f"  Median:   {np.median(all_num_frames):.0f}")
        print(f"  Total:    {sum(all_num_frames)}")
        
        if all_feature_means:
            print(f"\nFeature Mean Values Across Steps:")
            print(f"  Mean of means: {np.mean(all_feature_means):.4f}")
            print(f"  Std of means:  {np.std(all_feature_means):.4f}")
            print(f"  Min mean:      {np.min(all_feature_means):.4f}")
            print(f"  Max mean:      {np.max(all_feature_means):.4f}")
    
    def export_step(self, step_idx: int, output_path: str):
        """Export a single step to a separate file."""
        info = self.get_step_info(step_idx)
        
        np.savez_compressed(
            output_path,
            recording_id=info['recording_id'],
            step_id=info['step_id'],
            start_time=info['start_time'],
            end_time=info['end_time'],
            description=info['description'],
            has_errors=info['has_errors'],
            features=info['features'],
            num_frames=info['num_frames']
        )
        
        print(f"Exported step {step_idx} to {output_path}")
    
    def filter_steps(self, 
                    min_frames: Optional[int] = None,
                    max_frames: Optional[int] = None,
                    min_duration: Optional[float] = None,
                    max_duration: Optional[float] = None,
                    has_errors: Optional[bool] = None,
                    recording_id: Optional[str] = None):
        """Filter steps based on criteria."""
        filtered_indices = []
        
        for idx in range(self.num_steps):
            info = self.get_step_info(idx)
            duration = info['end_time'] - info['start_time']
            
            # Apply filters
            if min_frames is not None and info['num_frames'] < min_frames:
                continue
            if max_frames is not None and info['num_frames'] > max_frames:
                continue
            if min_duration is not None and duration < min_duration:
                continue
            if max_duration is not None and duration > max_duration:
                continue
            if has_errors is not None and info['has_errors'] != has_errors:
                continue
            if recording_id is not None and info['recording_id'] != recording_id:
                continue
            
            filtered_indices.append(idx)
        
        print(f"\nFiltered Results: {len(filtered_indices)} steps")
        print(f"{'='*80}\n")
        
        for idx in filtered_indices:
            info = self.get_step_info(idx)
            duration = info['end_time'] - info['start_time']
            error_flag = "⚠️ " if info['has_errors'] else ""
            
            print(f"Step {idx:3d} | {error_flag}{info['recording_id']:10s} | "
                  f"{info['num_frames']:3d} frames | {duration:6.2f}s | "
                  f"{info['description'][:50]}")
        
        return filtered_indices


    def show_raw_keys(self, limit: int = 50):
        """Show raw underlying keys in the npz file."""
        print(f"\n{'='*80}")
        print(f"RAW NPZ FILE CONTENTS")
        print(f"{'='*80}")
        
        all_keys = list(self.data.keys())
        print(f"Total keys: {len(all_keys)}")
        
        print("\nFirst 50 keys:")
        for k in all_keys[:limit]:
            val = self.data[k]
            if isinstance(val, np.ndarray):
                print(f"  {k:45s} : Array of shape {val.shape}, dtype {val.dtype}")
            else:
                # Truncate string representations
                val_str = str(val)[:50] + ("..." if len(str(val)) > 50 else "")
                print(f"  {k:45s} : {type(val).__name__} = {val_str}")
        
        if len(all_keys) > limit:
            print(f"  ... and {len(all_keys) - limit} more keys")

def main():
    parser = argparse.ArgumentParser(
        description='Inspect and analyze video features per step',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Show summary statistics
  python3 inspect_step_features.py video_features_by_steps.npz --summary
  
  # List all steps
  python3 inspect_step_features.py video_features_by_steps.npz --list
  
  # Show raw underlying npz keys
  python3 inspect_step_features.py video_features_by_steps.npz --raw
        """
    )
    
    parser.add_argument('file_path', type=str, help='Path to the .npz file')
    parser.add_argument('--mode', type=str, choices=['combined', 'separate'], 
                       default='combined', help='File mode')
    
    # Actions
    parser.add_argument('--raw', action='store_true', 
                       help='Show raw layout and keys of the npz file')
    parser.add_argument('--summary', action='store_true', 
                       help='Show overall statistics summary')
    parser.add_argument('--list', action='store_true', 
                       help='List all steps')
    parser.add_argument('--list-detailed', action='store_true', 
                       help='List all steps with detailed information')
    parser.add_argument('--show', type=int, metavar='INDEX', 
                       help='Show detailed information about a specific step')
    parser.add_argument('--search', type=str, metavar='QUERY', 
                       help='Search for steps matching query')
    parser.add_argument('--field', type=str, 
                       choices=['description', 'recording_id', 'step_id'],
                       default='description', help='Field to search in')
    parser.add_argument('--compare', type=int, nargs='+', metavar='INDEX', 
                       help='Compare multiple steps')
    parser.add_argument('--export', type=int, metavar='INDEX', 
                       help='Export a step to a file')
    parser.add_argument('--output', type=str, 
                       help='Output path for export')
    
    # Filtering
    parser.add_argument('--filter', action='store_true', 
                       help='Filter steps based on criteria')
    parser.add_argument('--min-frames', type=int, 
                       help='Minimum number of frames')
    parser.add_argument('--max-frames', type=int, 
                       help='Maximum number of frames')
    parser.add_argument('--min-duration', type=float, 
                       help='Minimum duration in seconds')
    parser.add_argument('--max-duration', type=float, 
                       help='Maximum duration in seconds')
    parser.add_argument('--has-errors', type=lambda x: x.lower() == 'true',
                       help='Filter by error status (true/false)')
    parser.add_argument('--recording-id', type=str, 
                       help='Filter by recording ID')
    
    args = parser.parse_args()
    
    # Check if file exists
    if not Path(args.file_path).exists():
        print(f"Error: File not found: {args.file_path}")
        sys.exit(1)
    
    # Initialize inspector
    inspector = StepFeatureInspector(args.file_path, args.mode)
    
    # Execute actions
    action_taken = False

    if args.raw:
        inspector.show_raw_keys()
        action_taken = True
    
    if args.summary:
        inspector.get_statistics_summary()
        action_taken = True
    
    if args.list:
        inspector.list_all_steps(show_details=False)
        action_taken = True
    
    if args.list_detailed:
        inspector.list_all_steps(show_details=True)
        action_taken = True
    
    if args.show is not None:
        inspector.show_step(args.show)
        action_taken = True
    
    if args.search:
        inspector.search_steps(args.search, args.field)
        action_taken = True
    
    if args.compare:
        inspector.compare_steps(args.compare)
        action_taken = True
    
    if args.export is not None:
        if not args.output:
            print("Error: --output required when using --export")
            sys.exit(1)
        inspector.export_step(args.export, args.output)
        action_taken = True
    
    if args.filter:
        inspector.filter_steps(
            min_frames=args.min_frames,
            max_frames=args.max_frames,
            min_duration=args.min_duration,
            max_duration=args.max_duration,
            has_errors=args.has_errors,
            recording_id=args.recording_id
        )
        action_taken = True
    
    if not action_taken:
        print("No action specified. Use --help to see available options.")
        print("\nQuick start: use --summary to see overall statistics")


if __name__ == '__main__':
    main()
