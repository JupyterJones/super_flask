import os
import tempfile
from datetime import datetime
from PIL import Image
import moviepy.editor as mp

# Step 0: Clean up old residual temporary files
def cleanup_resized_images(root_dir):
    for subdir, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith("_resized.jpg"):
                try:
                    os.remove(os.path.join(subdir, file))
                    print(f"Deleted old temporary file: {file}")
                except Exception as e:
                    print(f"Error deleting file {file}: {e}")

# Step 1: Search Directories for Images
def find_images_in_directory(root_dir):
    image_extensions = ('.jpg', '.jpeg', '.png', '.gif')
    image_files = []
    
    for subdir, _, files in os.walk(root_dir):
        for file in files:
            if file.lower().endswith(image_extensions):
                image_files.append(os.path.join(subdir, file))
    
    return sorted(image_files)  # Sort for consistent ordering

# Step 2: Resize Images to 512x768 and Convert to RGB if Necessary
def resize_image(image_path, target_size=(512, 768)):
    with Image.open(image_path) as img:
        # Ensure image has 3 channels (RGB). If it's grayscale or has alpha, convert to RGB
        if img.mode != "RGB":
            img = img.convert("RGB")
        
        # Resize the image
        resized_img = img.resize(target_size, Image.LANCZOS)  
        return resized_img


# Step 3: Create a video from images in batches
def create_video_from_images(image_files, output_filename, fps=24, batch_size=100):
    clips = []
    batch_count = 0
    temp_clips = []
    
    # Process images in batches
    for i, image_file in enumerate(image_files):
        resized_image = resize_image(image_file)
        
        # Save resized image temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_img:
            resized_image.save(temp_img.name, "JPEG")
            temp_path = temp_img.name
        
        # Create moviepy image clip
        img_clip = mp.ImageClip(temp_path).set_duration(1 / fps)
        clips.append(img_clip)

        # Delete the temporary image file after loading it into moviepy
        os.remove(temp_path)
        
        # Process each batch
        if (i + 1) % batch_size == 0 or (i + 1) == len(image_files):
            print(f"Processing batch {batch_count + 1} with {len(clips)} images...")
            temp_video = mp.concatenate_videoclips(clips, method="compose")
            
            # Save the batch video to a temporary file
            temp_video_path = f"temp_video_batch_{batch_count}.mp4"
            temp_video.write_videofile(temp_video_path, fps=fps)
            
            # Store the temporary video path for final concatenation
            temp_clips.append(temp_video_path)
            
            # Clear memory by resetting the clips list
            clips = []
            batch_count += 1

    # Step 4: Concatenate all batch videos into a final video
    final_clips = [mp.VideoFileClip(batch) for batch in temp_clips]
    final_video = mp.concatenate_videoclips(final_clips, method="compose")
    
    # Write the final video file
    final_video.write_videofile(output_filename, fps=fps)

    # Clean up temporary batch video files
    for batch in temp_clips:
        if os.path.exists(batch):
            os.remove(batch)

# Main function to orchestrate the backup process
def create_backup_video(root_dir, fps=24, batch_size=100):
    # Step 0: Clean up any old temporary resized images
    cleanup_resized_images(root_dir)
    
    # Step 1: Find all images in the static directory
    images = find_images_in_directory(root_dir)
    
    if not images:
        print("No images found in the directory.")
        return
    
    # Step 2: Generate a filename for the backup video
    current_date = datetime.now().strftime("%Y-%m-%d")
    output_filename = f"ARCHIVE_{current_date}_.mp4"
    
    # Step 3: Create the video from the found images in batches
    create_video_from_images(images, output_filename, fps=fps, batch_size=batch_size)
    
    print(f"Backup video created successfully: {output_filename}")

if __name__ == "__main__":
    # Path to the root directory containing the images (e.g., 'static')
    static_directory = 'static'
    
    # Create the backup video from images
    create_backup_video(static_directory, fps=24, batch_size=100)
