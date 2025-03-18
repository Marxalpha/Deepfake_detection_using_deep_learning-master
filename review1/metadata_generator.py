import os
import csv

# Define dataset path and output CSV file
dataset_path = "Dataset/FF++"
output_csv = "global_metadata.csv"

# Ensure the dataset directory exists
if not os.path.exists(dataset_path):
    print(f"Error: Dataset directory '{dataset_path}' does not exist.")
    exit()

# Open CSV file for writing (no header)
with open(output_csv, mode="w", newline="") as file:
    writer = csv.writer(file)

    # Loop through both real and fake directories
    for label in ["real", "fake"]:
        folder_path = os.path.join(dataset_path, label)
        
        if not os.path.exists(folder_path):
            print(f"Warning: '{folder_path}' not found. Skipping...")
            continue
        
        # Iterate through video files in the directory
        for filename in os.listdir(folder_path):
            if filename.endswith(".mp4"):  # Process only MP4 files
                writer.writerow([filename, label.upper()])

print(f"Metadata CSV generated successfully: {output_csv}")
