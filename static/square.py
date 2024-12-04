import os
import random
from flask import Flask, render_template, request, send_from_directory
from PIL import Image
import moviepy.editor as mpy

app = Flask(__name__)

# Paths
SQUARE_FOLDER = 'static/square'
TEMP_FOLDER = 'static/temp_512x512'
VIDEO_OUTPUT = 'static/videos/515x768.mp4'
AUDIO_FILE = 'static/voice/ALL_VOICE.mp3'

# Ensure the temp folder exists
if not os.path.exists(TEMP_FOLDER):
    os.makedirs(TEMP_FOLDER)

def logit(message):
    # Your logging function (replace with your implementation)
    print(message)

# Step 1: Resize images to 512x512
def resize_square_images():
    logit("Resizing square images...")
    for filename in os.listdir(SQUARE_FOLDER):
        if filename.endswith(('.jpg', '.png')):
            img_path = os.path.join(SQUARE_FOLDER, filename)
            img = Image.open(img_path)
            img = img.resize((512, 512))
            output_path = os.path.join(TEMP_FOLDER, os.path.splitext(filename)[0] + '.jpg')
            img.save(output_path, 'JPEG')
            logit(f"Resized and saved: {output_path}")

# Step 2: Get resized images for selection
def get_resized_images():
    resize_square_images()
    return os.listdir(TEMP_FOLDER)

# Step 3: Display images for selection
@app.route('/')
def index():
    images = get_resized_images()
    logit(f"Available images for selection: {images}")
    return render_template('512x512.html', images=images)

# Serve resized images for the client
@app.route('/static/temp_512x512/<filename>')
def serve_image(filename):
    return send_from_directory(TEMP_FOLDER, filename)

# Step 4: Create scrolling video from selected images
def create_video(selected_images, duration=45.0):
    logit("Creating video from selected images...")
    image_clips = []

    for image in selected_images:
        image_path = os.path.join(TEMP_FOLDER, image)
        image_clip = mpy.ImageClip(image_path).set_duration(duration / len(selected_images))
        image_clips.append(image_clip)

    # Skip the last image clip if any
    if len(image_clips) > 1:  # Ensure there's more than one clip
        image_clips = image_clips[:-1]
        logit(f"Total image clips loaded: {len(image_clips)}")

        # Concatenate the clips
        final_clip = mpy.concatenate_videoclips(image_clips, method="compose")
        logit(f"Concatenated {len(image_clips)} image clips.")

        # Load audio
        audio = mpy.AudioFileClip(AUDIO_FILE)
        audio_duration = audio.duration
        audio_start = random.uniform(0, audio_duration - final_clip.duration)
        audio = audio.subclip(audio_start, audio_start + final_clip.duration)

        # Set audio to the video
        final_clip = final_clip.set_audio(audio)

        # Write the final video file
        final_clip.write_videofile(VIDEO_OUTPUT, fps=24)
        logit(f"Video created successfully: {VIDEO_OUTPUT}")

@app.route('/create_square_video', methods=['POST'])
def create_square_video():
    selected_images = request.form.getlist('images')
    duration = request.form.get('speed', 45)  # Default duration is 45 seconds
    logit(f"Selected images: {selected_images}")
    logit(f"Video duration set to: {duration} seconds")

    if selected_images:
        # Create a video with the selected images
        create_video(selected_images, duration=float(duration))

    return render_template('512x512.html', images=get_resized_images(), video_filename='515x768.mp4')

if __name__ == '__main__':
    app.run(debug=True)
# In this script, we have defined a series of functions to create a scrolling video from a set of square images. The process involves resizing the square images to 512x512 pixels, displaying the resized images for selection, and creating a video from the selected images.