#open_door.py
import moviepy.editor as mp
import numpy as np
from sys import argv

def resize_if_needed(clip, target_size=(512, 768)):
    """Resizes the clip if it is not already the target size."""
    if clip.size != target_size:
        print(f"Image size {clip.size} is not {target_size}, resizing...")
        return clip.resize(target_size)
    print(f"Image is already the correct size: {clip.size}")
    return clip

def slide_apart_animation(foreground_path, background_path, output_path, duration=3, fps=24):
    print("Loading images...")

    # Load foreground and background images
    foreground = mp.ImageClip(foreground_path)
    background = mp.ImageClip(background_path)

    # Verify and resize if needed
    print("Verifying and resizing images if necessary...")
    foreground = resize_if_needed(foreground)
    background = resize_if_needed(background)

    # Split the foreground into two halves
    print("Splitting the foreground image into two halves.")
    left_half = foreground.crop(x1=0, y1=0, x2=256, y2=768)
    right_half = foreground.crop(x1=256, y1=0, x2=512, y2=768)

    # Define the sliding effect
    def make_frame(t):
        # Calculate how much the images have moved by time 't'
        slide_distance = min(256, int(256 * t / duration))

        # Create a blank frame (RGB format)
        frame = np.zeros((768, 512, 3), dtype=np.uint8)

        # Add background image to the frame
        frame[:, :, :] = background.get_frame(t)

        # Add the left sliding half
        if slide_distance < 256:
            left_part_width = 256 - slide_distance
            frame[:, :left_part_width] = left_half.get_frame(t)[:, slide_distance:256]

        # Add the right sliding half
        if slide_distance < 256:
            right_part_width = 256 + slide_distance
            frame[:, right_part_width:] = right_half.get_frame(t)[:, :256 - slide_distance]

        return frame

    # Create a video clip from the frames generated by make_frame
    print("Creating the animation...")
    animation = mp.VideoClip(make_frame, duration=duration)

    # Set the fps and write the video file
    print(f"Writing the output to {output_path}")
    animation.write_videofile(output_path, fps=fps)

    print("Animation complete.")

if __name__ == "__main__":
    # Paths to the foreground and background images and output MP4
    foreground_image = argv[1]
    background_image = argv[2]
    output_video = "output_animation.mp4"

    # Create the sliding door animation
    slide_apart_animation(foreground_image, background_image, output_video)
