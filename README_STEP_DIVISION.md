# Video Features Step Division Scripts

This repository contains scripts to divide video features into steps according to step annotations.

## Overview

The scripts process video feature files (`.npz` format) and divide them into individual steps based on temporal annotations. The output can be either:
- **Combined mode**: Single `.npz` file containing all steps from all videos
- **Separate mode**: Individual `.npz` files for each video

## Files

- `divide_features_by_steps.py` - Main script to process and divide video features
- `read_step_features.py` - Simple helper script to read and display the processed features
- `inspect_step_features.py` - Comprehensive inspection tool for detailed analysis

## Requirements

```bash
pip install numpy
```

## Usage

### Process Features

**Combined Mode (Single Output File):**
```bash
python3 divide_features_by_steps.py --mode combined
```

This creates a single file `video_features_by_steps.npz` containing all steps from all videos.

**Separate Mode (Per-Video Files):**
```bash
python3 divide_features_by_steps.py --mode separate
```

This creates individual files in the `video_features_by_steps/` directory (e.g., `9_2_steps.npz`, `9_4_steps.npz`, etc.).

### Advanced Options

```bash
python3 divide_features_by_steps.py \
    --features_dir video_features \
    --annotations annotations/annotation_json/step_annotations.json \
    --output video_features_by_steps.npz \
    --output_dir video_features_by_steps \
    --mode combined \
    --fps 1.0
```

**Parameters:**
- `--features_dir`: Directory containing video feature files (default: `video_features`)
- `--annotations`: Path to step annotations JSON file (default: `annotations/annotation_json/step_annotations.json`)
- `--output`: Output file path for combined mode (default: `video_features_by_steps.npz`)
- `--output_dir`: Output directory for separate mode (default: `video_features_by_steps`)
- `--mode`: Output mode - `combined` or `separate` (default: `combined`)
- `--fps`: Frames per second (default: `1.0` for 1s_1s features)

### Read Processed Features

**For Combined Files (Simple Reader):**
```bash
python3 read_step_features.py video_features_by_steps.npz --mode combined
```

**For Separate Files (Simple Reader):**
```bash
python3 read_step_features.py video_features_by_steps/9_2_steps.npz --mode separate
```

### Inspect and Analyze Features (Advanced)

The `inspect_step_features.py` script provides comprehensive analysis tools:

**Show overall statistics:**
```bash
python3 inspect_step_features.py video_features_by_steps.npz --summary
```

**List all steps:**
```bash
python3 inspect_step_features.py video_features_by_steps.npz --list
```

**Show detailed information about a specific step:**
```bash
python3 inspect_step_features.py video_features_by_steps.npz --show 5
```

**Search for steps:**
```bash
# Search in descriptions
python3 inspect_step_features.py video_features_by_steps.npz --search "pour" --field description

# Search by recording ID
python3 inspect_step_features.py video_features_by_steps.npz --search "9_2" --field recording_id
```

**Compare multiple steps:**
```bash
python3 inspect_step_features.py video_features_by_steps.npz --compare 0 5 10
```

**Filter steps by criteria:**
```bash
# Steps with 100+ frames and duration < 150s
python3 inspect_step_features.py video_features_by_steps.npz --filter --min-frames 100 --max-duration 150

# Steps with errors from a specific recording
python3 inspect_step_features.py video_features_by_steps.npz --filter --has-errors true --recording-id "9_2"
```

**Export a single step:**
```bash
python3 inspect_step_features.py video_features_by_steps.npz --export 5 --output step_5_exported.npz
```

## Data Structure

### Combined Mode Output

The combined `.npz` file contains:
- `num_steps`: Total number of steps across all videos
- For each step `i` (formatted as `step_XXXXX`):
  - `step_XXXXX_recording_id`: Video recording ID
  - `step_XXXXX_step_id`: Step identifier
  - `step_XXXXX_start_time`: Start time in seconds
  - `step_XXXXX_end_time`: End time in seconds
  - `step_XXXXX_description`: Step description
  - `step_XXXXX_has_errors`: Boolean indicating if step has errors
  - `step_XXXXX_features`: Feature array (num_frames, feature_dim)
  - `step_XXXXX_num_frames`: Number of frames in the step

### Separate Mode Output

Each video's `.npz` file contains:
- `recording_id`: Video recording ID
- `num_steps`: Number of steps in this video
- For each step `i` (formatted as `step_XXX`):
  - `step_XXX_step_id`: Step identifier
  - `step_XXX_start_time`: Start time in seconds
  - `step_XXX_end_time`: End time in seconds
  - `step_XXX_description`: Step description
  - `step_XXX_has_errors`: Boolean indicating if step has errors
  - `step_XXX_features`: Feature array (num_frames, feature_dim)
  - `step_XXX_num_frames`: Number of frames in the step

## Example Python Code

### Load Combined Features

```python
import numpy as np

data = np.load('video_features_by_steps.npz', allow_pickle=True)
num_steps = int(data['num_steps'])

# Access a specific step
step_idx = 0
prefix = f"step_{step_idx:05d}"
recording_id = str(data[f"{prefix}_recording_id"])
features = data[f"{prefix}_features"]
description = str(data[f"{prefix}_description"])

print(f"Step {step_idx}: {description}")
print(f"Features shape: {features.shape}")
```

### Load Separate Features

```python
import numpy as np

data = np.load('video_features_by_steps/9_2_steps.npz', allow_pickle=True)
recording_id = str(data['recording_id'])
num_steps = int(data['num_steps'])

# Access a specific step
step_idx = 0
prefix = f"step_{step_idx:03d}"
features = data[f"{prefix}_features"]
description = str(data[f"{prefix}_description"])

print(f"Video {recording_id}, Step {step_idx}: {description}")
print(f"Features shape: {features.shape}")
```

## Statistics (Example Run)

From a sample run with 16 video files:
- **Total videos processed**: 16
- **Total steps extracted**: 169
- **Total frames**: 8,972
- **Average frames per step**: 53.09
- **Feature dimension**: 256

## Notes

- The script assumes features are sampled at 1 FPS (1 second per frame) based on the `_1s_1s` suffix in filenames
- Features are extracted based on start/end times in the annotations
- Steps with invalid time ranges (start >= end) are skipped
- Frame indices are computed as: `frame_idx = time * fps`

## Inspection Features

The `inspect_step_features.py` script provides:
- **Summary statistics**: Overview of all steps including duration, frame counts, and feature statistics
- **Step listing**: View all steps with basic or detailed information
- **Step details**: In-depth view of individual steps with feature statistics
- **Search**: Find steps by description, recording ID, or step ID
- **Comparison**: Compare multiple steps side-by-side
- **Filtering**: Filter steps by frame count, duration, error status, or recording ID
- **Export**: Save individual steps to separate files

## Example Analysis Workflow

```bash
# 1. Start with summary statistics
python3 inspect_step_features.py video_features_by_steps.npz --summary

# 2. List all steps to get an overview
python3 inspect_step_features.py video_features_by_steps.npz --list

# 3. Examine specific steps of interest
python3 inspect_step_features.py video_features_by_steps.npz --show 5

# 4. Find all "microwave" steps
python3 inspect_step_features.py video_features_by_steps.npz --search "microwave"

# 5. Filter long duration steps
python3 inspect_step_features.py video_features_by_steps.npz --filter --min-duration 100

# 6. Compare filtered results
python3 inspect_step_features.py video_features_by_steps.npz --compare 5 10 15
```
