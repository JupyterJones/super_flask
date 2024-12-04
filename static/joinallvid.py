import os
from moviepy.editor import VideoFileClip, concatenate_videoclips

# Set the directory where your videos are located
input_directory = 'static/temp_exp/'
output_file = 'static/image-archives/output_video.mp4'

# Set the desired resolution for the output videos
target_resolution = (512, 768)  # Width, height

def resize_and_concatenate_videos(input_directory, target_resolution, output_file):
    video_clips = []
    
    # Loop through all files in the input directory
    for filename in sorted(os.listdir(input_directory)):
        file_path = os.path.join(input_directory, filename)

        # Only process if it's a video file (you can adjust the extensions as needed)
        if filename.endswith(('.mp4', '.avi', '.mov', '.mkv')):
            print(f"Processing {filename}...")

            # Load the video
            video = VideoFileClip(file_path)
            
            # Resize the video
            resized_video = video.resize(newsize=target_resolution)
            
            # Append to the list of video clips
            video_clips.append(resized_video)
    
    # Concatenate all video clips into one
    final_video = concatenate_videoclips(video_clips)

    # Write the final concatenated video to file
    final_video.write_videofile(output_file, codec="libx264")

    print(f"Output video saved as {output_file}")

# Run the function
resize_and_concatenate_videos(input_directory, target_resolution, output_file)
