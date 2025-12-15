import os
import urllib.request
import subprocess
from PIL import Image

def download_file(url, filename):
    print(f"Downloading {filename}...")
    urllib.request.urlretrieve(url, filename)
    print("Done.")

def extract_frames(video_path, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Use ffmpeg to extract frames
    # Assuming ffmpeg is installed. If not, this will fail.
    cmd = [
        "ffmpeg",
        "-i", video_path,
        os.path.join(output_dir, "im%d.png")
    ]
    subprocess.run(cmd, check=True)

def create_dataset_structure(root_dir):
    # Create structure: root/sequences/00001/0001/im1.png ...
    seq_dir = os.path.join(root_dir, "sequences", "00001", "0001")
    if not os.path.exists(seq_dir):
        os.makedirs(seq_dir)
    return seq_dir

def main():
    # URL for Foreman QCIF (small, standard test video)
    # Using a reliable source like Xiph.org
    url = "https://media.xiph.org/video/derf/y4m/foreman_qcif.y4m"
    video_filename = "foreman_qcif.y4m"
    
    dataset_root = "sample_dataset"
    
    if not os.path.exists(video_filename):
        download_file(url, video_filename)
    
    target_dir = create_dataset_structure(dataset_root)
    
    print(f"Extracting frames to {target_dir}...")
    try:
        extract_frames(video_filename, target_dir)
        print("Dataset prepared successfully!")
        print(f"You can now run: python train.py --dataset {dataset_root}")
    except FileNotFoundError:
        print("Error: ffmpeg not found. Please install ffmpeg.")
    except subprocess.CalledProcessError:
        print("Error extracting frames.")

if __name__ == "__main__":
    main()
