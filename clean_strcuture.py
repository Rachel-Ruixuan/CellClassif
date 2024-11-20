import os
import shutil
import re

def clean_video_id(video_id):
    # Remove non-numeric characters
    return re.sub(r'[^\d]', '', video_id)

def reorganize_data(base_dir, cell_type):
    # Create output directory structure
    output_base = os.path.join("/scratch/rl4789/CellClassif/data/raw", cell_type)
    os.makedirs(output_base, exist_ok=True)

    if cell_type == "MDA":
        # Process MDA files
        for folder in os.listdir(base_dir):
            if folder.startswith("100MDA_"):
                video_id = folder.split("_")[1]
                source_dir = os.path.join(base_dir, folder, "fibroblasts,mda-10x exp")
                output_dir = os.path.join(output_base, f"100MDA_{video_id}")
                os.makedirs(output_dir, exist_ok=True)

                # Copy and rename files if needed
                for file in os.listdir(source_dir):
                    if file.endswith("_ch00.jpg"):
                        shutil.copy2(
                            os.path.join(source_dir, file),
                            os.path.join(output_dir, f"100MDA_{video_id}_{file.split('_')[-2]}_{file.split('_')[-1]}")
                        )

    elif cell_type == "FB":
        # Process FB files
        for file in os.listdir(base_dir):
            if file.endswith("_ch00.jpg"):
                # Extract and clean video ID
                match = re.search(r'100FB[(\s]*(\d+)', file)
                if match:
                    video_id = clean_video_id(match.group(1))
                    output_dir = os.path.join(output_base, f"100FB_{video_id}")
                    os.makedirs(output_dir, exist_ok=True)
                    
                    # Create new filename
                    new_filename = f"100FB_{video_id}_{file.split('_')[-2]}_{file.split('_')[-1]}"
                    shutil.copy2(
                        os.path.join(base_dir, file),
                        os.path.join(output_dir, new_filename)
                    )


import os
import shutil
from pathlib import Path

def move_images_to_parent():
    # Base directory path
    base_dir = '/scratch/rl4789/CellClassif/data/raw/M2'
    
    # Get all subdirectories in M2
    m2_subdirs = [d for d in os.listdir(base_dir) 
                  if os.path.isdir(os.path.join(base_dir, d)) and d.startswith('M2a-')]
    
    for subdir in m2_subdirs:
        # Full path to the M2a-XXXXX directory
        dir_path = os.path.join(base_dir, subdir)
        
        # Path to Mark_and_Find 001 directory
        mark_and_find_dir = os.path.join(dir_path, 'Mark_and_Find 001')
        
        # Check if Mark_and_Find 001 directory exists
        if os.path.exists(mark_and_find_dir):
            print(f"Processing {subdir}...")
            
            # Get all files in Mark_and_Find 001
            try:
                files = os.listdir(mark_and_find_dir)
                
                # Move each file to parent directory
                for file in files:
                    src = os.path.join(mark_and_find_dir, file)
                    dst = os.path.join(dir_path, file)
                    
                    try:
                        shutil.move(src, dst)
                        print(f"Moved: {file}")
                    except Exception as e:
                        print(f"Error moving {file}: {str(e)}")
                
                # Remove the empty Mark_and_Find 001 directory
                try:
                    os.rmdir(mark_and_find_dir)
                    print(f"Removed empty directory: {mark_and_find_dir}")
                except Exception as e:
                    print(f"Error removing directory {mark_and_find_dir}: {str(e)}")
                    
            except Exception as e:
                print(f"Error processing directory {subdir}: {str(e)}")
        else:
            print(f"Mark_and_Find 001 directory not found in {subdir}")
            
if __name__ == "__main__":
    # Ask for confirmation before proceeding
    response = input("This script will move all files from 'Mark_and_Find 001' directories to their parent directories. Proceed? (y/n): ")
    
    if response.lower() == 'y':
        move_images_to_parent()
        print("Operation completed.")
    else:
        print("Operation cancelled.")

# # Example usage
# mda_base = "/scratch/rl4789/CellClassif/data/raw/MDA"
# reorganize_data(mda_base, "MDA")