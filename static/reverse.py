import subprocess
from random import randint

def reverse_video():
    inc = str(randint(0,999))
    input_video = '/home/jack/Desktop/Flask_Make_Art/static/video_resources/66ef4e6f0cf7d6beb7f4daad.mp4'
    output_video = '/home/jack/Desktop/Flask_Make_Art/static/video_resources'
    video_output_dir = 'static/temp_exp'
    # Define the FFmpeg command
    _cmd = [
        'ffmpeg',
        '-hide_banner',
        '-i', f'{input_video}',
        '-vf', 'reverse',
        '-af', 'areverse', 
        '-y', f'static/temp_exp/concatenated_videoX.mp4'
        ]
        
    # Start the FFmpeg process
    subprocess.run(_cmd, check=True)
    return "completed"

if __name__ == '__main__':
    reverse_video()        
