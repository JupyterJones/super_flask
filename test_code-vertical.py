from moviepy.editor import *

def make_scrolling_video(image_path, output_video_path, video_duration=10, video_size=(512, 768)):
    """Creates a video by scrolling across the image vertically."""
    
    print(f"Loading image from {image_path}")
    
    image = ImageClip(image_path)

    # Check if the image dimensions are valid
    if image.size[0] != video_size[0]:
        print("Error: Image width must be equal to the video width.")
        return

    # Function to create a scrolling effect vertically
    def scroll_func(get_frame, t):
        # Calculate the y position for scrolling
        y = int((image.size[1] - video_size[1]) * t / video_duration)
        return get_frame(t)[y:y + video_size[1], 0:video_size[0]]

    # Create the video clip with the scrolling effect
    video = VideoClip(lambda t: scroll_func(image.get_frame, t), duration=video_duration)
    video = video.set_fps(24)

    print(f"Saving video to {output_video_path}")
    video.write_videofile(output_video_path, codec='libx264', audio=False)

if __name__ == '__main__':
    scroll_speed = 10  # Duration for the scroll (total video length in seconds)
    image_path = 'static/temp_512x512/long_image.jpg'  # Your image path
    output_path = 'static/videos/515x768.mp4'  # Your output video path
    make_scrolling_video(image_path, output_path, scroll_speed)
