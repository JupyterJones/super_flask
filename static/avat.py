import os
import random
import shutil
import uuid
from moviepy.editor import ImageClip, AudioFileClip
from PIL import Image
import subprocess
from flask import Flask, render_template, request, redirect, url_for, jsonify 

app = Flask(__name__)

# Define static directories
AUDIO_DIR = 'static/audio_mp3'
IMAGE_DIR = 'static/archived_resources'
OUTPUT_DIR = 'static/temp_exp'

@app.route('/', methods=['GET'])
def render_add_sound_form():
    # Get list of audio and image files
    audio_files = [f for f in os.listdir(AUDIO_DIR) if f.endswith('.mp3')]
    image_files = [f for f in os.listdir(IMAGE_DIR) if f.endswith(('.png', '.jpg'))]

    # Shuffle and limit to 40 images
    random.shuffle(image_files)
    image_files = image_files[:40]

    return render_template('mk_avatar.html', audio_files=audio_files, image_files=image_files)

@app.route('/mk_avatar', methods=['POST'])
def mk_avatar_route():
    # Get the image and audio file paths from the form data
    image_file = request.form.get('image_path')  # Name of the select field in the form
    audio_file = request.form.get('audio')  # Name of the select field in the form

    if image_file and audio_file:
        # Construct full paths
        image_path = os.path.join(IMAGE_DIR, image_file)
        audio_path = os.path.join(AUDIO_DIR, audio_file)
        dest1 ='/home/jack/Desktop/Flask_Make_Art/Wav2Lip-master/sample_data/use.jpg'
        dest2 ='/home/jack/Desktop/Flask_Make_Art/Wav2Lip-master/sample_data/use.mp3'
        shutil.copy(image_path, dest1)
        shutil.copy(audio_path, dest2)
        # Process the image and audio to create a video
        output_path = add_sound_to_image(image_path, audio_path)

        if output_path:
            # Generate a unique ID for the output file
            uid = str(uuid.uuid4())
            final_output_path = f'{OUTPUT_DIR}/{uid}_avatar.mp4'

            shutil.copy(output_path, final_output_path)

            return render_template('mk_avatar.html', video=final_output_path)
        else:
            return "Error creating video", 500
    else:
        return "Missing image or audio path", 400

def add_sound_to_image(image_path, audio_path):
    try:
        # Load the image and create an ImageClip
        image_clip = ImageClip(image_path)

        # Load the audio clip
        audio_clip = AudioFileClip(audio_path)

        # Get the duration of the audio clip
        audio_duration = audio_clip.duration

        # Set the duration for the image clip (max duration 45 seconds)
        final_duration = min(audio_duration, 45)
        image_clip = image_clip.set_duration(final_duration)

        # Set the frames per second (fps) for the image clip
        image_clip.fps = 24

        # Set the audio to the image clip
        video_clip = image_clip.set_audio(audio_clip)

        # Define output path for the video
        output_path = image_path.replace('.jpg', '_avatar.mp4').replace('.png', '_avatar.mp4')

        # Write the video with audio to a new file
        video_clip.write_videofile(output_path, codec='libx264', audio_codec='aac')

        return output_path
    except Exception as e:
        print(f"Error creating video: {e}")
        return None
@app.route('/makeit')
def makeit():
    subprocess.run(['/home/jack/miniconda3/envs/cloned_base/bin/python', 'Wav2Lip-master/makeit'], check=True)
    return redirect(url_for('mk_videos'))

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5300)
