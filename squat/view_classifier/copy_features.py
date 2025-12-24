import os
import shutil
from pathlib import Path

# --- Configuration ---
VIDEO_ROOT = "./squat/view_classifier/videos"  # 5 folders with videos
SOURCE_JSON = "./squat/extracted_features"      # Flat folder with all JSONs
OUTPUT_JSON = "./squat/view_classifier/extracted_features"  # Mirror structure

def copy_features():
    """
    Reads video names from view_classifier/videos (in 5 subfolders),
    finds matching JSON files in extracted_features (flat structure),
    and copies them to view_classifier/extracted_features (same 5 folder structure)
    """
    
    if not os.path.exists(VIDEO_ROOT):
        print(f"ERROR: Video root not found: {VIDEO_ROOT}")
        return
    
    if not os.path.exists(SOURCE_JSON):
        print(f"ERROR: Source JSON folder not found: {SOURCE_JSON}")
        return
    
    # Track statistics
    found = 0
    missing = 0
    missing_files = []
    
    print("--- Scanning video folders ---")
    
    # Walk through each subfolder in videos directory
    for root, dirs, files in os.walk(VIDEO_ROOT):
        # Get relative path from VIDEO_ROOT (this gives us the subfolder name)
        rel_path = os.path.relpath(root, VIDEO_ROOT)
        
        # Skip the root directory itself
        if rel_path == ".":
            continue
        
        # Create corresponding output directory
        output_dir = os.path.join(OUTPUT_JSON, rel_path)
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"\nProcessing folder: {rel_path}")
        print(f"  Videos found: {len([f for f in files if f.endswith(('.mp4', '.avi', '.mov'))])}")
        
        # Process each video file
        for file in files:
            if file.lower().endswith(('.mp4', '.avi', '.mov')):
                # Get video name without extension
                video_name = os.path.splitext(file)[0]
                
                # Look for corresponding JSON in source folder
                json_filename = f"{video_name}.json"
                source_json_path = os.path.join(SOURCE_JSON, json_filename)
                
                if os.path.exists(source_json_path):
                    # Copy to output directory
                    dest_json_path = os.path.join(output_dir, json_filename)
                    shutil.copy2(source_json_path, dest_json_path)
                    found += 1
                else:
                    missing += 1
                    missing_files.append(f"{rel_path}/{video_name}")
        
        print(f"  JSONs copied: {found}")
    
    # Summary
    print(f"\n{'='*50}")
    print(f"✓ SUMMARY:")
    print(f"  - Found and copied: {found} JSON files")
    print(f"  - Missing: {missing} JSON files")
    
    if missing_files:
        print(f"\n⚠️  Missing JSON files (first 10):")
        for mf in missing_files[:10]:
            print(f"    - {mf}.json")
        
        if len(missing_files) > 10:
            print(f"    ... and {len(missing_files) - 10} more")
        
        print(f"\n💡 TIP: Run '2_extract_features.py' to generate missing features")
    
    print(f"\n✓ Output folder: {os.path.abspath(OUTPUT_JSON)}")

if __name__ == "__main__":
    copy_features()