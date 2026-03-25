import numpy as np
import argparse
import sys

def inspect_npz(file_path, max_videos=None):
    print(f"=== Inspecting file: {file_path} ===")
    try:
        data = np.load(file_path, allow_pickle=True)
        keys = list(data.keys())
        print(f"Total keys found: {len(keys)}")
        
        if max_videos is not None:
            keys = keys[:max_videos]
            print(f"Limiting to first {max_videos} keys.\n")
        else:
            print(f"Keys found: {keys}\n")
        
        for key in keys:
            print(f"--- Key: '{key}' ---")
            item = data[key]
            print(f"Type: {type(item)}")
            if isinstance(item, np.ndarray):
                print(f"Shape: {item.shape}")
                print(f"Data type: {item.dtype}")
                # Print a small sample if it's an array
                if item.size > 0:
                    flat_sample = item.flatten()
                    print(f"Sample data (first 5 elements): {flat_sample[:5]}")
                    if np.issubdtype(item.dtype, np.number):
                        # Calculate some basic statistics for numeric features
                        print(f"Statistics   : Min={np.min(item):.4f}, Max={np.max(item):.4f}, Mean={np.mean(item):.4f}")
            else:
                print(f"Value: {item}")
            print()
    except Exception as e:
        print(f"Error loading file: {e}", file=sys.stderr)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect the contents of a video features .npz file.")
    parser.add_argument("file_path", type=str, help="Path to the .npz file to inspect")
    parser.add_argument("--max_videos", type=int, default=None, help="Maximum number of videos (keys) to analyze")
    args = parser.parse_args()
    
    inspect_npz(args.file_path, args.max_videos)
