import json
import os
from datetime import datetime
from PIL import Image, UnidentifiedImageError, ImageFile
import moviepy.editor as mp

# Step 1: Search Directories for Images
def find_images_in_directory(root_dir):
    image_extensions = ('.jpg', '.jpeg', '.png', '.gif')
    image_files = []
    
    for subdir, _, files in os.walk(root_dir):
        for file in files:
            if file.lower().endswith(image_extensions):
                if '_resized.jpg' not in file.lower():  # Skip already resized images
                    image_files.append(os.path.join(subdir, file))    
    return sorted(image_files)  # Sort for consistent ordering

# Ensure that PIL can handle truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Step 2: Resize Images
def resize_image(image_path, target_size=(512, 768)):
    try:
        with Image.open(image_path) as img:
            if img.mode != "RGB":
                img = img.convert("RGB")  # Ensure RGB format
            resized_img = img.resize(target_size, Image.LANCZOS)
            # Return a copy of the resized image to avoid closure issues
            return resized_img.copy()  
    except UnidentifiedImageError:
        print(f"Error: Cannot identify image file {image_path}. Skipping.")
        return None  # Skip the image if it's unreadable
    except ValueError as e:
        print(f"Error resizing image {image_path}: {e}")
        return None  # Skip the image if there's a resize error


# Step 3: Create video segment from a batch of images
def create_video_segment(image_batch, segment_index, source_directory_name, fps=24):
    clips = []
    temp_files = []
    
    for image_file in image_batch:
        resized_image = resize_image(image_file)
        if resized_image:
            temp_path = image_file + "_resized.jpg"
            resized_image.save(temp_path, "JPEG")
            temp_files.append(temp_path)
            img_clip = mp.ImageClip(temp_path).set_duration(1 / fps)
            clips.append(img_clip)

    if clips:
        # Create the segment filename based on the source directory name
        segment_filename = f"{source_directory_name}-segment_{segment_index}.mp4"
        print(f"Creating segment video: {segment_filename}")  # Debug statement
        video = mp.concatenate_videoclips(clips, method="compose")
        
        try:
            video.write_videofile(segment_filename, fps=fps)
            print(f"Segment video created: {segment_filename}")  # Debug statement
        except Exception as e:
            print(f"Error creating segment video: {e}")

        # Clean up the temporary resized images
        for temp_file in temp_files:
            if os.path.exists(temp_file):
                os.remove(temp_file)
                print(f"Deleted temporary file: {temp_file}")
        
        return segment_filename  # Return the path of the created segment file
    return None

# Step 4: Write segment information to a JSON file
def update_video_segments_json(segment_info, json_path):
    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            existing_data = json.load(f)
    else:
        existing_data = {"segments": []}
    
    existing_data["segments"].append(segment_info)

    with open(json_path, 'w') as f:
        json.dump(existing_data, f, indent=4)

# Step 5: Concatenate all segments into the final video
def concatenate_segments(segment_files, output_filename):
    if segment_files:
        print(f"Concatenating segments into final video: {output_filename}")  # Debug statement
        video_clips = [mp.VideoFileClip(segment) for segment in segment_files]
        final_video = mp.concatenate_videoclips(video_clips, method="compose")
        
        try:
            final_video.write_videofile(output_filename, fps=24)
            print(f"Final video created: {output_filename}")  # Debug statement
        except Exception as e:
            print(f"Error creating final video: {e}")
        
        # Clean up the segment files
        for segment in segment_files:
            if os.path.exists(segment):
                #os.remove(segment)
                print(f"Deleted segment file: {segment}")
    else:
        print("No segments to concatenate.")

def create_backup_video(root_dir, fps=24, batch_size=500):
    print(f"Current Working Directory: {os.getcwd()}")  # Debug statement
    images = find_images_in_directory(root_dir)
    
    if not images:
        print("No images found in the directory.")
        return
    
    source_directory_name = os.path.basename(root_dir)
    json_path = f"{source_directory_name}-new_video_segments.json"

    # Load existing segments from JSON if it exists
    existing_segments = []
    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            existing_data = json.load(f)
            existing_segments = [segment['segment_path'] for segment in existing_data.get("segments", [])]

    # Step 2: Process images in batches and create video segments
    segment_files = []
    for i in range(0, len(images), batch_size):
        segment_index = i // batch_size
        segment_file_name = f"{source_directory_name}-segment_{segment_index}.mp4"
        
        # Check if the segment has already been created
        if segment_file_name in existing_segments:
            print(f"Segment {segment_file_name} already processed. Skipping...")
            segment_files.append(segment_file_name)
            continue

        image_batch = images[i:i + batch_size]
        segment_file = create_video_segment(image_batch, segment_index, source_directory_name, fps=fps)
        if segment_file:
            segment_files.append(segment_file)
            segment_info = {
                "source_directory_name": source_directory_name,
                "source_directory": root_dir,
                "segment_path": segment_file,
                "segment_index": segment_index,
                "image_batch": image_batch,
                "segment": os.path.basename(segment_file),
                "directory": root_dir,
                "batch_start": i,
                "batch_end": min(i + batch_size, len(images))
            }
            update_video_segments_json(segment_info, json_path)

    # Step 3: Concatenate all video segments into the final video
    current_date = datetime.now().strftime("%Y-%m-%d")
    output_filename = f"{source_directory_name}-ARCHIVE({current_date}).mp4"
    concatenate_segments(segment_files, output_filename)

if __name__ == "__main__":
    # Set the path to the directory containing the images
    static_directory = '/mnt/HDD500/collections/images'
    
    # Create the backup video from images
    create_backup_video(static_directory, fps=24, batch_size=500)
