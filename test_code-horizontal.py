from moviepy.editor import *
import os
def make_scrolling_video(image_path, output_video_path, video_duration=10, video_size=(512, 768)):
    """Creates a video by scrolling across the image from left to right."""
    
    print(f"Loading image from {image_path}")
    
    image = ImageClip(image_path)

    def scroll_func(get_frame, t):
        x = int((image.size[0] - video_size[0]) * t / video_duration)
        return get_frame(t)[0:video_size[1], x:x+video_size[0]]
    
    video = VideoClip(lambda t: scroll_func(image.get_frame, t), duration=video_duration)
    video = video.set_fps(24)

    print(f"Saving video to {output_video_path}")
    video.write_videofile(output_video_path, codec='libx264', audio=False)
if __name__ == '__main__':
    scroll_speed = 100  # Adjust scroll speed in pixels per second
    image_path='static/temp_512x512/long_image.jpg'
    output_path = 'static/videos/515x768.mp4'
    make_scrolling_video(image_path, output_path, scroll_speed)
