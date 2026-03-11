# Step Feature Inspection Tools - Quick Reference

## 🎯 Quick Start

```bash
# 1. Process video features into steps
python3 divide_features_by_steps.py --mode combined

# 2. Get summary statistics
python3 inspect_step_features.py video_features_by_steps.npz --summary

# 3. Explore specific steps
python3 inspect_step_features.py video_features_by_steps.npz --show 5
```

## 📋 Available Scripts

| Script | Purpose | Use When |
|--------|---------|----------|
| `divide_features_by_steps.py` | Process and divide video features | Initial data preparation |
| `read_step_features.py` | Simple reader for quick viewing | Quick inspection of output files |
| `inspect_step_features.py` | Comprehensive analysis tool | Detailed analysis and filtering |
| `example_inspection_usage.py` | Programming examples | Integrating into your code |

## 🔍 Common Inspection Tasks

### View Overall Statistics
```bash
python3 inspect_step_features.py video_features_by_steps.npz --summary
```
**Shows:** Total steps, duration stats, frame counts, feature statistics

### List All Steps
```bash
# Simple list
python3 inspect_step_features.py video_features_by_steps.npz --list

# Detailed list
python3 inspect_step_features.py video_features_by_steps.npz --list-detailed
```

### Examine Specific Steps
```bash
# Single step with full statistics
python3 inspect_step_features.py video_features_by_steps.npz --show 5

# Compare multiple steps
python3 inspect_step_features.py video_features_by_steps.npz --compare 0 5 10 15
```

### Search for Steps
```bash
# By description (case-insensitive)
python3 inspect_step_features.py video_features_by_steps.npz --search "microwave"

# By recording ID
python3 inspect_step_features.py video_features_by_steps.npz --search "9_2" --field recording_id

# By step ID
python3 inspect_step_features.py video_features_by_steps.npz --search "99" --field step_id
```

### Filter Steps
```bash
# Long duration steps (> 100s)
python3 inspect_step_features.py video_features_by_steps.npz --filter --min-duration 100

# Steps with many frames (100-150)
python3 inspect_step_features.py video_features_by_steps.npz --filter --min-frames 100 --max-frames 150

# Short steps from specific recording
python3 inspect_step_features.py video_features_by_steps.npz --filter --recording-id "9_2" --max-duration 30

# Steps with errors
python3 inspect_step_features.py video_features_by_steps.npz --filter --has-errors true

# Steps without errors
python3 inspect_step_features.py video_features_by_steps.npz --filter --has-errors false
```

### Export Steps
```bash
# Export single step
python3 inspect_step_features.py video_features_by_steps.npz --export 5 --output step_5.npz
```

## 📊 Output Information

### Summary Statistics Include:
- Total steps, unique recordings, unique step IDs
- Duration statistics (mean, std, min, max, median)
- Frame counts per step
- Feature value statistics

### Step Details Include:
- Recording ID and Step ID
- Time range and duration
- Description
- Error status
- Number of frames
- Feature array shape
- Feature statistics (mean, std, min, max, percentiles)
- Per-frame statistics

## 💡 Analysis Examples

### Find all "microwave" steps over 1 minute
```bash
python3 inspect_step_features.py video_features_by_steps.npz --search "microwave" | grep -E "^Step" > microwave_steps.txt
python3 inspect_step_features.py video_features_by_steps.npz --filter --min-duration 60
```

### Identify outlier steps (very short or very long)
```bash
# Very short (< 10s)
python3 inspect_step_features.py video_features_by_steps.npz --filter --max-duration 10

# Very long (> 150s)
python3 inspect_step_features.py video_features_by_steps.npz --filter --min-duration 150
```

### Analyze a specific recording in detail
```bash
# Filter steps from one recording
python3 inspect_step_features.py video_features_by_steps.npz --filter --recording-id "9_2"

# Or use separate mode file
python3 inspect_step_features.py video_features_by_steps/9_2_steps.npz --mode separate --list-detailed
```

### Compare similar steps across videos
```bash
# Find all "pour" steps
python3 inspect_step_features.py video_features_by_steps.npz --search "pour"

# Then compare specific ones
python3 inspect_step_features.py video_features_by_steps.npz --compare 7 42 83
```

## 🐍 Programmatic Usage

See `example_inspection_usage.py` for complete examples including:
- Basic step information retrieval
- Batch analysis across all steps
- Custom filtering logic
- Feature extraction for machine learning
- Temporal pattern analysis

```python
from inspect_step_features import StepFeatureInspector

# Initialize
inspector = StepFeatureInspector('video_features_by_steps.npz', mode='combined')

# Get step info
info = inspector.get_step_info(0)
print(f"Step 0: {info['description']}")
print(f"Features shape: {info['features'].shape}")

# Iterate through all steps
for i in range(inspector.num_steps):
    info = inspector.get_step_info(i)
    # Your analysis here
```

## 🎓 Tips

1. **Start with `--summary`** to understand your dataset
2. **Use `--list`** to browse available steps
3. **Use `--filter`** to narrow down to interesting subsets
4. **Use `--compare`** to analyze similar steps side-by-side
5. **Combine with grep/awk** for advanced text processing
6. **Export interesting steps** for focused analysis

## 📁 File Formats

### Combined File (`video_features_by_steps.npz`)
- Contains all steps from all videos
- Step keys: `step_00000`, `step_00001`, etc.
- Includes: `num_steps` metadata

### Separate Files (`video_features_by_steps/RECORDING_ID_steps.npz`)
- One file per video
- Step keys: `step_000`, `step_001`, etc.
- Includes: `recording_id`, `num_steps` metadata

### Exported Step Files
- Single step with all metadata
- Direct keys: `recording_id`, `step_id`, `features`, etc.
- Can be loaded directly with `np.load()`
