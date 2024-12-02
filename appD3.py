#!/home/jack/miniconda3/envs/cloned_base/bin/python3
# -*- coding: utf-8 -*-
import os
import random
from random import randint
import time
import signal
import subprocess
import shutil
import string
import uuid
import html
import re
import glob
import datetime
import inspect
import psutil
from sys import argv
import moviepy.editor as mp
from moviepy.video.fx import crop
from moviepy.video.fx.fadein import fadein
from moviepy.video.fx.fadeout import fadeout
# Flask Imports
from flask import (
    Flask, request, render_template, redirect, url_for, send_from_directory, 
    send_file, flash, jsonify, make_response, Response, session, abort, 
    render_template_string, after_this_request,Blueprint, abort
)
from werkzeug.utils import secure_filename
from werkzeug.middleware.profiler import ProfilerMiddleware
from flask_caching import Cache
from flask_cors import CORS


# Image Processing Libraries
from PIL import (
    Image, ImageOps, ImageDraw, ImageFont, ImageFilter, ImageEnhance, 
    ImageSequence, ImageChops, ImageStat, ImageColor, ImagePalette
)
import cv2
import dlib
from skimage import future, data, segmentation, filters, color, io
from skimage.future import graph
from skimage.io import imread, imsave
import matplotlib.pyplot as plt

# Video and Audio Processing (MoviePy, ffmpeg, gTTS, Pygame)
from moviepy.editor import (
    ImageClip, VideoClip, clips_array, concatenate_videoclips, CompositeVideoClip, 
    ColorClip, VideoFileClip, AudioFileClip, concatenate_audioclips, TextClip, 
    ImageSequenceClip
)

from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
from gtts import gTTS
import pygame

# Machine Learning Libraries
import tensorflow as tf
import numpy as np
import tensorflow_hub as hub
from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()

# External Libraries
import yt_dlp
import threading
# Custom Logging
import datetime
import os
import weakref
import wave  
from balacoon_tts import TTS  
from pydub import AudioSegment  
from pydub.playback import play
import base64
import io
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import sqlite3
from icecream import ic
from io import BytesIO 
import gc
from pytrends.request import TrendReq
import time
import threading

cache = weakref.WeakValueDictionary()

def add_to_cache(key, value):
    cache[key] = value

def get_from_cache(key):
    return cache.get(key)


#gc.set_debug(gc.DEBUG_LEAK)    
app = Flask(__name__)

CORS(app)

#app.register_blueprint(file_manager_bp)
# Configuring SimpleCache with a limit on cache size
app.config['CACHE_TYPE'] = 'SimpleCache'
app.config['CACHE_DEFAULT_TIMEOUT'] = 300  # 5 minutes cache timeout
app.config['CACHE_THRESHOLD'] = 100  # Max 100 items in cache
app.config['MAX_CONTENT_LENGTH'] = 20 * 1024 * 1024  # 20 MB limit

cache = Cache(app)

# Define the log file path
LOG_FILE_PATH = 'static/app_log.txt'

# Ensure the log file exists or create it
if not os.path.exists(LOG_FILE_PATH):
    with open(LOG_FILE_PATH, 'w'):
        pass  # Create an empty log file if it doesn't exist

# Logging function
def logit(message):
    try:
        # Get the current timestamp
        timestr = datetime.datetime.now().strftime('%A_%b-%d-%Y_%H-%M-%S')

        # Get the caller's frame information
        caller_frame = inspect.stack()[1]
        filename = caller_frame.filename
        lineno = caller_frame.lineno

        # Convert message to string if it's a list
        if isinstance(message, list):
            message_str = ' '.join(map(str, message))
        else:
            message_str = str(message)

        # Construct the log message with filename and line number
        log_message = f"{timestr} - File: {filename}, Line: {lineno}: {message_str}\n"

        # Open the log file in append mode
        with open(LOG_FILE_PATH, "a") as file:
            # Write the log message to the file
            file.write(log_message)

        # Print the log message to the console
        #print(log_message)

    except Exception as e:
        # If an exception occurs during logging, print an error message
        print(f"Error occurred while logging: {e}")
   
@app.route('/readlog')
def readlog():
    logdatas = open(LOG_FILE_PATH, "r").read().split("\n")
    logit(logdatas)
    return render_template('read_log.html', log_content=logdatas)
# Logging function

def readlog():
    log_file_path = 'static/app_log.txt'    
    with open(log_file_path, "r") as Input:
        logdata = Input.read()
    # print last entry
    logdata = logdata.split("\n")
    return logdata
@app.route('/delete_log')
def delete_log():
    open(LOG_FILE_PATH, "w").close()
    logit("Log file deleted successfully")
    return redirect('/view_log')
# Logging function
@app.route('/view_log', methods=['GET', 'POST'])
def view_log():
    data = readlog()
    return render_template('view_log.html', data=data)
#app.config['PROFILE'] = True
#app.wsgi_app = ProfilerMiddleware(app.wsgi_app, restrictions=[30])

UPLOAD_FOLDER = 'static/novel_images'
app.config['UPLOAD_FOLDER'] = 'static/novel_images'
MASK_FOLDER = 'static/masks'
app.config['MASK_FOLDER'] = 'static/archived-masks'
NOVEL_IMAGES = 'static/novel_images'
app.config['NOVEL_IMAGES'] = 'static/novel_images'
app.config['MAX_CONTENT_LENGTH'] = 20 * 1024 * 1024  # Limit upload size to 16MB
ARCHIVED_IMAGES_FOLDER = 'static/archived-images'
app.config['ARCHIVED_IMAGES_FOLDER'] = 'static/archived-images'
DOWNLOAD_FOLDER = 'static/downloads'
app.config['DOWNLOAD_FOLDER'] = 'static/downloads'
UPLOADFOLDER = 'static/novel_images'
app.config['UPLOADFOLDER'] = 'static/novel_images'
ARCHIVED_RESOURCES = 'static/archived_resources'
app.config['ARCHIVED_RESOURCES'] = 'static/archived_resources'
# Ensure the directories exist
app.config['NOVEL_IMAGES'] = 'static/novel_images'
os.makedirs(DOWNLOAD_FOLDER, exist_ok=True)
os.makedirs(ARCHIVED_IMAGES_FOLDER, exist_ok=True)
os.makedirs(ARCHIVED_RESOURCES, exist_ok=True)
os.makedirs(UPLOADFOLDER, exist_ok=True)
os.makedirs(MASK_FOLDER, exist_ok=True)
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif','webp','mp4','mkv','avi'}
# Configurations
app.config['TEMP_FOLDER'] = 'static/temp/'
app.config['FONT_FOLDER'] = 'static/fonts/'
app.config['NOVEL_IMAGES'] = 'static/novel_images/'
TEMPLATE_DIR = 'templates'
app.config['TEMPLATE_DIR'] = 'templates'
os.makedirs(app.config['TEMP_FOLDER'], exist_ok=True)
currentDir = os.getcwd()

@app.route('/')
@cache.cached(timeout=60)
def index():
    session['visited_index'] = True
    image_paths = stored_image_paths()
    post = get_intro(limit=1)
    decoded_post = []
    for row in post:
        # Replace newlines in the content (third field) with <br>
        id, title, content, image, video_filename = row
        if content:
            content = content.replace('\r\n', '<br>').replace('\n', '<br>')  # handle both \r\n and \n
            decoded_post.append((id, title, content, image, video_filename))
    return render_template('index.html', post=decoded_post)

@app.before_request
def ensure_index_page():
    # Check if the session is new (i.e., user hasn't visited the index page)
    if 'visited_index' not in session:
        # If the requested URL is not the index or static files, redirect to the index page
        if request.endpoint != 'index' and not request.path.startswith('/static'):
            return redirect(url_for('index'))       
@app.route('/favicons.ico')
def favicons():
    return send_from_directory(os.path.join(app.root_path, 'static'), 'favicon.ico', mimetype='image/vnd.microsoft.icon')
@app.route('/favicon.ico')
def favicon():
    # Set the size of the favicon
    size = (16, 16)

    # Create a new image with a transparent background
    favicon = Image.new('RGBA', size, (0, 0, 0, 0))

    # Create a drawing object
    draw = ImageDraw.Draw(favicon)

    # Draw a yellow square
    square_color = (255, 0, 255)
    draw.rectangle([(0, 0), size], fill=square_color)
    circle_color = (255, 0, 0) 
    # Draw a red circle
    circle_center = (size[0] // 2, size[1] // 2)
    circle_radius = size[0] // 3
    draw.ellipse(
        [(circle_center[0] - circle_radius, circle_center[1] - circle_radius),
         (circle_center[0] + circle_radius, circle_center[1] + circle_radius)],
        fill=circle_color
    )

    # Save the image to a memory buffer
    image_buffer = io.BytesIO()
    favicon.save(image_buffer, format='ICO')
    image_buffer.seek(0)

    return Response(image_buffer.getvalue(), content_type='image/x-icon')
        
#@app.route('/convert', methods=['POST'])
def convert_images():
    # Directory containing the JPG images   
    # Check if directory exists
    if not os.path.isdir(app.config['NOVEL_IMAGES']):
        return redirect(url_for('index'))
    
    # Loop through all files in the directory
    image_directory = app.config['NOVEL_IMAGES']
    for filename in os.listdir(image_directory):
        if filename.lower().endswith('.jpg'):
            # Construct full file paths
            jpg_path = os.path.join(image_directory, filename)
            png_filename = os.path.splitext(filename)[0] + '.png'
            png_path = os.path.join(image_directory, png_filename)
            
            try:
                # Open the JPG image and convert it to PNG
                with Image.open(jpg_path) as img:
                    img = img.convert('RGBA')  # Ensure image has alpha channel if needed
                    img.save(png_path, format='PNG')
                
                # Remove the original JPG file
                os.remove(jpg_path)
                
            except Exception as e:
                print(f"Failed to convert {jpg_path}. Error: {e}")
    return redirect(url_for('index'))

@app.route('/mk_mask')
def mk_mask():
    masks=glob.glob('static/archived-images/mask*.jpg')
    # list by date, last image first
    masks = sorted(masks, key=os.path.getmtime, reverse=True)
    filenames = [os.path.basename(mask) for mask in masks]
    mask_data = zip(masks, filenames)
    return render_template('mk_mask.html',mask_data=mask_data)


@app.route('/create_circle_mask', methods=['POST'])
def create_circle_mask():
    # Get input values from the form
    x = int(request.form.get('x', 0))
    y = int(request.form.get('y', 0))
    size = int(request.form.get('size', 50)) + 20
    feather = int(request.form.get('feather', 20))
    aspect = int(request.form.get('aspect', 0))
    
    # Calculate width and height based on aspect
    if aspect > 0:
        width = size + aspect  # Make width larger for wide aspect
        height = size
    elif aspect < 0:
        width = size
        height = size + abs(aspect)  # Make height larger for tall aspect
    else:
        width, height = size, size  # Default to square aspect ratio

    # Create a black background image (size 512x768)
    background = Image.new('RGBA', (512, 768), (0, 0, 0, 255))
    
    # Create a white ellipse (or circle if width == height)
    ellipse = Image.new('RGBA', (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(ellipse)
    draw.ellipse((0, 0, width, height), fill=(255, 255, 255, 255))

    # Apply feathering (blur the edges)
    ellipse = ellipse.filter(ImageFilter.GaussianBlur(feather))

    # Calculate position to paste the ellipse (centered by default)
    paste_position = (256 + x - width // 2, 384 + y - height // 2)
    background.paste(ellipse, paste_position, ellipse)
    background = background.convert('RGB')
    
    # Optionally blur the whole background
    background = background.filter(ImageFilter.GaussianBlur(30))
    
    # Save the result in static/archived-images
    mask_path = f'static/archived-images/mask_{x}_{y}_{size}_{feather}_{aspect}.jpg'
    background.save(mask_path)
    # save a copy of the mask to static/masks with same name as the mask_path not mask png
    shutil.copy(mask_path, 'static/masks/' + os.path.basename(mask_path))
     
    # List and sort masks by date (latest first)
    masks = glob.glob('static/archived-images/mask*.jpg')
    masks = sorted(masks, key=os.path.getmtime, reverse=True)
    filenames = [os.path.basename(mask) for mask in masks]
    mask_data = zip(masks, filenames)

    return render_template('mk_mask.html', mask_path=mask_path, mask_data=mask_data)

@app.route('/create_rectangle_mask', methods=['POST'])
def create_rectangle_mask():
    # Get input values from the form
    x = int(request.form.get('x', 0))
    y = int(request.form.get('y', 0))
    size = int(request.form.get('size', 50)) + 20
    feather = int(request.form.get('feather', 20))
    aspect = int(request.form.get('aspect', 0))

    # Calculate width and height based on aspect
    if aspect > 0:
        width = size + aspect  # Make width larger for wide aspect
        height = size
    elif aspect < 0:
        width = size
        height = size + abs(aspect)  # Make height larger for tall aspect
    else:
        width, height = size, size  # Default to square aspect ratio

    # Create a new background image with a transparent background (RGBA)
    background = Image.new('RGBA', (512, 768), (0, 0, 0, 255))

    # Create a rectangle image with a transparent background (RGBA)
    rectangle_image = Image.new('RGBA', (width, height), (0, 0, 0, 0))

    # Create an ImageDraw object to draw on the rectangle image
    draw = ImageDraw.Draw(rectangle_image)

    # Draw a solid white rectangle (change color as needed)
    draw.rectangle([0, 0, width, height], fill=(255, 255, 255, 255))

    # Apply feathering (blur the edges of the rectangle)
    rectangle_image = rectangle_image.filter(ImageFilter.GaussianBlur(feather))

    # Calculate the position to paste the rectangle (centered by default)
    paste_position = (256 + x - width // 2, 384 + y - height // 2)

    # Paste the rectangle onto the background with alpha blending
    background.paste(rectangle_image, paste_position, rectangle_image)

    # Optionally blur the whole background if needed
    background = background.filter(ImageFilter.GaussianBlur(30))

    # Convert the image to RGB mode
    background = background.convert('RGB')

    # Save the result in static/archived-images
    mask_path = f'static/archived-images/mask_{x}_{y}_{size}_{feather}_{aspect}.jpg'
    background.save(mask_path)

    # Save a copy of the mask to static/masks with the same name
    shutil.copy(mask_path, 'static/masks/' + os.path.basename(mask_path))

    # List and sort masks by date (latest first)
    masks = glob.glob('static/archived-images/mask*.jpg')
    masks = sorted(masks, key=os.path.getmtime, reverse=True)
    filenames = [os.path.basename(mask) for mask in masks]
    mask_data = zip(masks, filenames)

    return render_template('mk_mask.html', mask_path=mask_path, mask_data=mask_data)







    #return send_file(mask_path, as_attachment=True)
def save_text_to_file(filename, text):
    try:
        with open(os.path.join(TEXT_FILES_DIR, filename), "w") as file:
            file.write(text)
    except Exception as e:
        print(f"An error occurred while saving file '{filename}': {e}")

# Route for the form
@app.route('/add_text', methods=['GET', 'POST'])
def add_text():
    # Get the list of images in the NOVEL_IMAGES
    images = os.listdir(app.config['NOVEL_IMAGES'])
    images = [img for img in images if img.lower().endswith(('.png', '.jpg', '.jpeg'))]   
    
    if request.method == 'POST':
        image_file = request.form['image_file']
        text = request.form['text']
        position = (int(request.form['x_position']), int(request.form['y_position']))
        font_size = int(request.form['font_size'])
        color = request.form['color']
        font_path = os.path.join(app.config['FONT_FOLDER'], 'xkcd-script.ttf')
        font = ImageFont.truetype(font_path, font_size)

        # Open the image
        image_path = os.path.join(app.config['NOVEL_IMAGES'], image_file)
        image = Image.open(image_path)

        # Draw the text on the image
        draw = ImageDraw.Draw(image)
        draw.text(position, text, font=font, fill=color)

        # Save the temporary image for preview
        temp_image_path = os.path.join(app.config['NOVEL_IMAGES'], 'temp-image.png')
        image.save(temp_image_path)

        return render_template('add_text.html', images=images, selected_image=image_file, temp_image='temp-image.png', text=text, position=position, font_size=font_size, color=color)
    
    return render_template('add_text.html', images=images)


# Route to save the final image
@app.route('/save_image', methods=['POST'])
def save_image():
    image_file = request.form['image_file']
    final_text = request.form['final_text']
    position = eval(request.form['final_position'])  # Convert string back to tuple
    font_size = int(request.form['final_font_size'])
    color = request.form['final_color']
    font_path = os.path.join(app.config['FONT_FOLDER'], 'xkcd-script.ttf')
    font = ImageFont.truetype(font_path, font_size)

    # Open the image again
    image_path = os.path.join(app.config['NOVEL_IMAGES'], image_file)
    image = Image.open(image_path)

    # Draw the final text on the image
    draw = ImageDraw.Draw(image)
    draw.text(position, final_text, font=font, fill=color)

    # Save the image with a unique UUID
    unique_filename = f"{uuid.uuid4()}.png"
    final_image_path = os.path.join(app.config['NOVEL_IMAGES'], unique_filename)
    image.save(final_image_path)


    return redirect(url_for('add_text'))

if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

if not os.path.exists(app.config['MASK_FOLDER']):
    os.makedirs(app.config['MASK_FOLDER'])

if not os.path.exists(app.config['NOVEL_IMAGES']):
    os.makedirs(app.config['NOVEL_IMAGES'])





def stored_image_paths():
    image_paths = []
    for ext in ['png', 'jpg', 'jpeg']:
        image_paths.extend(glob.glob(os.path.join(app.config['NOVEL_IMAGES'], f'*.{ext}')))
    image_paths = sorted(image_paths, key=os.path.getmtime, reverse=True)
    return image_paths


@app.route('/mk_videos')
def mk_videos():
    image_paths = stored_image_paths()
    return render_template('mk_videos.html', image_paths=image_paths)
@app.route('/img_processing')
def img_processing_route():
    image_paths = stored_image_paths()
    #sorted by date, last image first
    image_paths = sorted(image_paths, key=os.path.getmtime, reverse=True)
    return render_template('img_processing.html', image_paths=image_paths)
def load_images(image_directory):
    image_paths = []
    for ext in ['png', 'jpg', 'jpeg']:
        image_paths.extend(glob.glob(os.path.join(image_directory, f'*.{ext}')))
    random.shuffle(image_paths)
    #image_paths = sorted(image_paths, key=os.path.getmtime, reverse=True)
    return image_paths[:3]
def load_image(image_directory):
    image_paths_ = []
    for ext in ['png', 'jpg', 'jpeg']:
        image_paths_.extend(glob.glob(os.path.join(image_directory, f'*.{ext}')))
    #random.shuffle(image_paths)
    image_paths_ = sorted(image_paths_, key=os.path.getmtime, reverse=True)
    return image_paths_[:3]
def convert_to_grayscale(image_path):
    image = Image.open(image_path).convert('L')
    mask_path = os.path.join(app.config['MASK_FOLDER'], 'greyscale_mask.png')
    image.save(mask_path)
    #copy to upload folder
    shutil.copy(mask_path, app.config['NOVEL_IMAGES'])
    shutil.copy(mask_path, 'static/archived-images')

    return mask_path

def convert_to_binary(image_path):
    # Convert image to grayscale
    image = Image.open(image_path).convert('L')
    
    # Calculate the mean pixel value to use as the threshold
    np_image = np.array(image)
    threshold = np.mean(np_image)
    
    # Convert image to binary based on the mean threshold
    binary_image = image.point(lambda p: 255 if p > threshold else 0)
    
    # Save the binary mask
    mask_path = os.path.join(app.config['MASK_FOLDER'], 'binary_mask.png')
    binary_image.save(mask_path)
    
    # Invert the binary mask
    inverted_image = binary_image.point(lambda p: 255 - p)
    
    # Save the inverted binary mask
    inverted_mask_path = os.path.join(app.config['MASK_FOLDER'], 'inverted_binary_mask.png')
    inverted_image.save(inverted_mask_path)
    
    # Copy both images to the upload folder
    shutil.copy(mask_path, 'static/archived-images')
    shutil.copy(inverted_mask_path, 'static/archived-images')
    
    return mask_path, inverted_mask_path

def resize_images_to_base(base_image, images):
    base_size = base_image.size
    resized_images = [base_image]
    for img in images[1:]:
        resized_images.append(img.resize(base_size, resample=Image.Resampling.LANCZOS))
    return resized_images

@app.route('/get_images', methods=['POST','GET'])
def get_images():
    image_directory = app.config['UPLOAD_FOLDER']
    image_paths = load_images(image_directory)
    #shuffle the images
    random.shuffle(image_paths)
    image_paths_ = load_image(image_directory)
    random.shuffle(image_paths_)
    return render_template('display_images_exp.html', image_paths=image_paths, mask_path=None, opacity=0.5, image_paths_=image_paths_)


@app.route('/play_narration', methods=['GET', 'POST'])
def play_narration():
    music = 'static/audio/narration.mp3'
    return render_template('play_mp3.html', music=music)
@app.route('/edit_mask', methods=['POST'])
def edit_mask():
    image_paths = request.form.getlist('image_paths')
    mask_path = request.form.get('mask_path')
    opacity = float(request.form.get('opacity', 0.5))
    return render_template('display_images_exp.html', image_paths=image_paths, mask_path=mask_path, opacity=opacity)

@app.route('/store_result', methods=['POST'])
def store_result():
    result_image_path = request.form.get('result_image')
    unique_id = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    store_path = os.path.join(app.config['NOVEL_IMAGES'], f'result_{unique_id}.png')
    
    # Correct the path for the result image
    result_image_path = result_image_path.replace('/static/', 'static/')
    
    # Save the result image to the store folder
    image = Image.open(result_image_path)
    image.save(store_path)
    return redirect(url_for('index'))

@app.route('/refresh-images')
def refresh_images():
    try:
        # Run the script using subprocess
        subprocess.run(['/home/jack/miniconda3/envs/cloned_base/bin/python', 'refresh_images.py'], check=True)
        return redirect(url_for('index'))
    except subprocess.CalledProcessError as e:
        return f"An error occurred: {e}'"

@app.route('/refresh-video')
def refresh_video():
    try:
        convert_images_route()    
        createvideo()
        subprocess.run(['/home/jack/miniconda3/envs/cloned_base/bin/python', 'refresh_video.py'], check=True)
        subprocess.run(['/home/jack/miniconda3/envs/cloned_base/bin/python', 'Best_FlipBook'], check=True)
        subprocess.run(['/home/jack/miniconda3/envs/cloned_base/bin/python', 'diagonal_transition'], check=True)
        subprocess.run(['/home/jack/miniconda3/envs/cloned_base/bin/python', 'blendem'], check=True)
        subprocess.run(['/bin/bash', 'slide'], check=True)         
        subprocess.run(['/bin/bash', 'zoomX4'], check=True)
        subprocess.run(['/home/jack/miniconda3/envs/cloned_base/bin/python', 'fadem'], check=True)
        subprocess.run(['/bin/bash', 'zoomY4'], check=True)
        subprocess.run(['/home/jack/miniconda3/envs/cloned_base/bin/python', 'vertical_scroll'], check=True)
        video_path = 'static/temp_exp/diagonal1.mp4'
        add_title(video_path, hex_color="#A52A2A")
        
        subprocess.run(['/home/jack/miniconda3/envs/cloned_base/bin/python', 'joinvid'], check=True)

        return redirect(url_for('create_video'))
    
    except subprocess.CalledProcessError as e:
        return f"An error occurred: {e}"

@app.route('/display_resources', methods=['POST', 'GET'])
def display_resources():
    # Logging function for debugging
    logit('display_resources function called')
    
    # Glob pattern to get image paths in static/archived_resource directory
    image_paths = glob.glob('static/archived_resources/*.jpg') + glob.glob('static/archived_resource/*.png')+ glob.glob('static/archived_resources/*.jpeg')
    logit(f'Image paths found: {image_paths}')
    
    # Sort images by modification date, newest first
    image_paths = sorted(image_paths, key=os.path.getmtime, reverse=True)
    logit(f'Sorted image paths: {image_paths}')
    
    # Render template and pass image paths
    return render_template('display_resources_exp.html', image_paths=image_paths)

@app.route('/copy_images', methods=['GET', 'POST'])
def copy_images():
    size_and_format_images_route()
    if request.method == 'POST':
        selected_images = request.form.getlist('selected_images')
        
        # Copy the selected images to the store folder
        for image_path in selected_images:
            logit(f'___Copying image: {image_path}')
            shutil.copy(image_path, 'static/novel_resources')
            shutil.copy(image_path, 'static/novel_images')
        
        # Redirect to a page where you can view the stored images
        return redirect(url_for('img_processing_route'))
 
@app.route('/select_mask_image', methods=['POST', 'GET'])
def select_mask_image():
    if request.method == 'POST':
        selected_image = request.form.get('selected_image')
        if not selected_image:
            return "Please select an image for masking."
        return render_template('choose_mask.html', selected_image=selected_image)
    image_paths = get_image_paths()
    return render_template('select_mask_image.html', image_paths=image_paths)
@app.route('/choose_mask', methods=['POST'])
def choose_mask():
    selected_image = request.form.get('selected_image')
    mask_type = request.form.get('mask_type')

    if not selected_image:
        return "Please select an image for masking."
    
    if mask_type == 'grayscale':
        mask_path = convert_to_grayscale(selected_image)
    elif mask_type == 'binary':
        mask_path = convert_to_binary(selected_image)
    else:
        return "Invalid mask type selected."
    #redirect to select
    return redirect(url_for('select_images'))
#render_template('select_images.html', image_paths=[selected_image], mask_path=mask_path, opacity=0.5)


def extract_random_frames(video_path, num_frames=25):
    try:
        video = VideoFileClip(video_path)
        duration = video.duration
        timestamps = sorted([random.uniform(0, duration) for _ in range(num_frames)])
        saved_images = []
        
        for i, timestamp in enumerate(timestamps):
            frame = video.get_frame(timestamp)
            img = Image.fromarray(frame)
            image_filename = f"frame_{i+1}.jpg"
            image_path = os.path.join(app.config['ARCHIVED_IMAGES_FOLDER'], image_filename)
            img.save(image_path)
            saved_images.append(image_filename)
        
        return saved_images
    except Exception as e:
        print(f"Error extracting frames: {e}")
        raise

@app.route('/get_video_images', methods=['GET', 'POST'])
def get_video_images():
    if request.method == 'POST':
        url = request.form['url']
        if url:
            try:
                # Download the YouTube video
                video_path = download_youtube_video(url)
                
                # Extract 25 random frames
                images = extract_random_frames(video_path)
                
                # Redirect to display the images
                return redirect(url_for('display_images'))
            except Exception as e:
                print(f"Error in /get_images: {e}")
                return str(e)
    
    return render_template('get_images.html')

@app.route('/images')
def display_images():
    try:
        images = os.listdir(app.config['ARCHIVED_IMAGES_FOLDER'])
        images = [os.path.join(app.config['ARCHIVED_IMAGES_FOLDER'], img) for img in images]
        return render_template('YouTube_gallery.html', images=images)
    except Exception as e:
        print(f"Error in /images: {e}")
        return str(e)

def download_youtube_video(url):
    try:
        # Set the download options
        ydl_opts = {
            'outtmpl': os.path.join(app.config['DOWNLOAD_FOLDER'], '%(title)s.%(ext)s'),
            'format': 'mp4',  # Best format available
            'noplaylist': True
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            # Extract video information and download the video
            info_dict = ydl.extract_info(url, download=True)
            video_title = info_dict.get('title')
            
            # Sanitize and format the filename to remove spaces and special characters
            sanitized_title = secure_filename(video_title)
            sanitized_title = sanitized_title.replace(" ", "_")
            download_path = os.path.join(app.config['DOWNLOAD_FOLDER'], f"{sanitized_title}.mp4")
            static_video_path = os.path.join('static', 'temp.mp4')

            # Find the downloaded file
            for root, dirs, files in os.walk(app.config['DOWNLOAD_FOLDER']):
                for file in files:
                    if file.endswith('.mp4'):
                        actual_downloaded_file = os.path.join(root, file)
                        break
                else:
                    continue
                break

            # Check if the video was downloaded correctly
            if os.path.exists(actual_downloaded_file):
                # Move the downloaded video to the static/temp.mp4 path
                shutil.move(actual_downloaded_file, static_video_path)
            else:
                raise FileNotFoundError(f"File not found: {actual_downloaded_file}")

            return static_video_path

    except Exception as e:
        print(f"Error downloading video: {e}")
        raise
def create_feathered_image(foreground_path, output_path):
    # Load the foreground image
    foreground = cv2.imread(foreground_path)
    height, width = foreground.shape[:2]

    # Initialize dlib's face detector
    detector = dlib.get_frontal_face_detector()
    gray_foreground = cv2.cvtColor(foreground, cv2.COLOR_BGR2GRAY)
    faces = detector(gray_foreground)

    # Create an alpha channel and a binary mask
    alpha_channel = np.zeros((height, width), dtype=np.uint8)

    if len(faces) == 0:
        print("No face detected in the image. Using the entire image with no feathering.")
        # Use the entire image with a full alpha channel
        alpha_channel = np.full((height, width), 255, dtype=np.uint8)
    else:
        for face in faces:
            x, y, w, h = face.left(), face.top(), face.width(), face.height()
            center = (x + w // 2, y + h // 2)
            radius = max(w, h) // 2
            cv2.circle(alpha_channel, center, radius, 255, -1)

        # Feather the edges of the mask
        alpha_channel = cv2.GaussianBlur(alpha_channel, (101, 101), 0)

    # Add the alpha channel to the foreground image
    foreground_rgba = np.dstack((foreground, alpha_channel))

    # Save the result as a PNG file with transparency
    cv2.imwrite(output_path, foreground_rgba)

    print(f"Feathered image saved to: {output_path}")

    return output_path

def overlay_feathered_on_background(foreground_path, background_path, output_path):
    # Load the feathered image and background image
    foreground = cv2.imread(foreground_path, cv2.IMREAD_UNCHANGED)  # Load with alpha channel
    background = cv2.imread(background_path)

    # Resize and crop both images to 512x768
    foreground = resize_and_crop(foreground)
    background = resize_and_crop(background)

    # Extract the alpha channel from the foreground image
    alpha_channel = foreground[:, :, 3] / 255.0
    foreground_rgb = foreground[:, :, :3]

    # Ensure background has 4 channels
    background_rgba = cv2.cvtColor(background, cv2.COLOR_BGR2BGRA)
    background_alpha = background_rgba[:, :, 3] / 255.0

    # Validate dimensions
    if foreground_rgb.shape[:2] != background_rgba.shape[:2]:
        raise ValueError(f"Foreground and background dimensions do not match: {foreground_rgb.shape[:2]} vs {background_rgba.shape[:2]}")

    # Blend the images
    for i in range(3):  # For each color channel
        background_rgba[:, :, i] = (foreground_rgb[:, :, i] * alpha_channel + background_rgba[:, :, i] * (1 - alpha_channel)).astype(np.uint8)

    # Save the result
    cv2.imwrite(output_path, background_rgba)

    print(f"Composite image saved to: {output_path}")

    im = Image.open(output_path).convert('RGB')
    im.save(output_path[:-3] + 'jpg', quality=95)
    return output_path

def resize_and_crop(image, target_width=512, target_height=768):
    # Resize the image to fit the target dimensions while maintaining the aspect ratio
    height, width = image.shape[:2]
    aspect_ratio = width / height
    target_aspect_ratio = target_width / target_height

    if aspect_ratio > target_aspect_ratio:
        new_width = target_width
        new_height = int(new_width / aspect_ratio)
    else:
        new_height = target_height
        new_width = int(new_height * aspect_ratio)

    resized_image = cv2.resize(image, (new_width, new_height))

    # Ensure the resized image is at least the target dimensions
    if resized_image.shape[0] < target_height or resized_image.shape[1] < target_width:
        resized_image = cv2.resize(image, (target_width, target_height))

    # Crop the resized image to the target dimensions
    crop_x = (resized_image.shape[1] - target_width) // 2
    crop_y = (resized_image.shape[0] - target_height) // 2
    cropped_image = resized_image[crop_y:crop_y + target_height, crop_x:crop_x + target_width]

    return cropped_image

@app.route('/face_detect', methods=['POST', 'GET'])
def face_detect():
    if request.method == 'POST':
        # Check if the file part is in the request
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        
        file = request.files['file']

        # Handle case where no file is selected
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)

        # Validate file and save if allowed
        if file and allowed_file(file.filename):
            # Secure the filename and save the file
            filename = secure_filename(file.filename)
            save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(save_path)

            # Create a feathered PNG image from the detected face
            feathered_image_path = create_feathered_image(save_path, 'static/archived-images/feathered_face.png')
            # Randomly select a background image from the static/archived-images/ folder
            background_image_path = random.choice(glob.glob("static/archived-images/*.jpg"))

            # Overlay the feathered image on the background and generate the composite image
            output_composite_path = overlay_feathered_on_background(feathered_image_path, background_image_path, 'static/archived-images/composite_image.png')

            # Generate unique filenames using UUID for feathered and composite images
            feathered_image_uuid = str(uuid.uuid4()) + '.png'
            composite_image_uuid = str(uuid.uuid4()) + '.png'

            # Copy the feathered and composite images to the archived_resources directory
            feathered_image_archive_path = os.path.join('static/archived_resources', feathered_image_uuid)
            composite_image_archive_path = os.path.join('static/archived_resources', composite_image_uuid)

            shutil.copy(feathered_image_path, feathered_image_archive_path)
            shutil.copy(output_composite_path, composite_image_archive_path)

            # Return the template with paths to the generated images
            return render_template('face_detect.html', feathered_image=feathered_image_archive_path, composite_image=composite_image_archive_path)

    # Render the face_detect template on GET request or error cases
    return render_template('face_detect.html')

@app.route('/about')#, methods=['POST', 'GET'])
def about():
    return render_template('application_overview.html')
def resize_image(image_path):
    # Open the image
    image = Image.open(image_path)
    # Resize the image
    resized_image = image.resize((512, 768), Image.LANCZOS)
    # Save the resized image
    resized_image.save(image_path)
    print(f"Resized image saved at: {image_path}")
    
@app.route('/resize_all')#, methods=['POST', 'GET'])
def resize_all():
    # Resize all images in the upload folder
    image_paths = glob.glob(os.path.join(app.config['UPLOAD_FOLDER'], '*.jpg'))
    for image_path in image_paths:
        resize_image(image_path)
    return redirect(url_for('index'))


@app.route('/upload_image', methods=['GET', 'POST'])
def upload_image():
    if request.method == 'POST':
        image_file = request.files['image']
        image_file = convert_webp_to_jpg(image_file)
        if image_file:
            filename = image_file.filename
            image_path = os.path.join('static', filename)  # Ensure only one 'static/' prefix
            image_file.save(image_path)
            return render_template('confirm_image.html', image_path=filename)  # Pass only the filename
    return render_template('upload_image.html')

@app.route('/torn_edge', methods=['POST'])
def create_torn_edge_effect():
    filename = request.form.get('image_path')
    image_path = os.path.join('static', filename)
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    output_filename = f"result_{timestamp}.png"
    output_path = os.path.join('static', 'NOVEL_IMAGES', output_filename)
    image = Image.open(image_path).convert("RGBA")
    width, height = image.size

    # Create a mask with the same size as the image
    mask = Image.new("L", (width, height), 255)
    draw = ImageDraw.Draw(mask)

    # Create a random torn edge effect for all edges
    np.random.seed(0)
    torn_edge_top = np.random.normal(0, 60, width)
    torn_edge_bottom = np.random.normal(0, 60, width)
    torn_edge_left = np.random.normal(0, 60, height)
    torn_edge_right = np.random.normal(0, 60, height)

    torn_edge_top = np.clip(torn_edge_top, -100, 100)
    torn_edge_bottom = np.clip(torn_edge_bottom, -100, 100)
    torn_edge_left = np.clip(torn_edge_left, -100, 100)
    torn_edge_right = np.clip(torn_edge_right, -100, 100)

    # Apply torn edges to the top and bottom
    for x in range(width):
        draw.line((x, 0, x, int(torn_edge_top[x])), fill=0)
        draw.line((x, height, x, height - int(torn_edge_bottom[x])), fill=0)

    # Apply torn edges to the left and right
    for y in range(height):
        draw.line((0, y, int(torn_edge_left[y]), y), fill=0)
        draw.line((width, y, width - int(torn_edge_right[y]), y), fill=0)

    # Apply Gaussian blur to smooth the edges
    mask = mask.filter(ImageFilter.GaussianBlur(2))

    result = Image.composite(image, Image.new("RGBA", image.size, (255, 255, 255, 0)), mask)
    result.save(output_path, "PNG")
    #copy to upload folder
    shutil.copy(output_path, app.config['UPLOAD_FOLDER'])
    return render_template('torn_edge.html', original_image=filename, torn_image=os.path.join('NOVEL_IMAGES', output_filename))


SCAN_DATA_FILE = 'local_scan_data.json'
IMAGE_EXTENSIONS = ['.jpg', '.jpeg', '.png','webp']
STATIC_IMAGE_DIR = 'static/local_images'
STATIC_IMAGE_DIR = 'static/local_images'
if not os.path.exists(STATIC_IMAGE_DIR):
    os.makedirs(STATIC_IMAGE_DIR)
STATIC_GALLERY_DATA_FILE = 'local_static_gallery_data.json'

app.static_folder = 'static'  # Set the static folder to 'static'
app.config['SECRET_KEY'] = 'your_secret_key'
app.config['VUPLOAD_FOLDER'] = 'static/videos'
app.config['ALLOWED_EXTENSIONS'] = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif','webp','mp4','mkv','avi'}
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif','webp','mp4','mkv','avi'}

#app.config['ALLOWED_EXTENSIONS'] = {'mp4'}
#app.config['DATABASE'] = 'static/blog3.db'
app.config['DATABASE'] = 'static/blog4.db'
DATABASE = app.config['DATABASE']
app.config['DATABASEF'] = 'static/functions.db'
DATABASEF = app.config['DATABASEF']

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/upload_video/<int:post_id>', methods=['POST'])
def upload_video(post_id):
    if 'videoFile' not in request.files:
        print('No file part')
        flash('No file part')
        return redirect(request.url)
    
    file = request.files['videoFile']
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['VUPLOAD_FOLDER'], filename))
        print(f"Filename: {filename}")
        # Update the database with the filename
        update_video_filename(post_id, filename)
        flash('Video uploaded successfully')
        return redirect(url_for('post', post_id=post_id))
    else:
        flash('Allowed file types are .mp4')
        return redirect(request.url)

def update_video_filename(post_id, filename):
    with sqlite3.connect(DATABASE) as conn:
        cursor = conn.cursor()
        cursor.execute('UPDATE post SET video_filename = ? WHERE id = ?', (filename, post_id))
        conn.commit()

'''
# Initialize SQLite database if not exists
def init_db():
    with sqlite3.connect(DATABASE) as conn:
        cursor = conn.cursor()
        cursor.execute(''
            CREATE TABLE IF NOT EXISTS post (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT UNIQUE,
                content TEXT NOT NULL,
                video_filename TEXT NULL,
                image BLOB
            )
        '')
        conn.commit()
'''
# Function to fetch a single post by ID
@app.route('/post/<int:post_id>', methods=['GET', 'POST'])
def post(post_id):
    if request.method == 'POST':
        return upload_video(post_id)

    post = get_post(post_id)
    if not post:
        flash('Post not found')
        return redirect(url_for('home'))

    image_data = get_image_data(post_id)
    video_filename = post[4] if post[4] else None  # Adjust index based on your database schema

    return render_template('post.html', post=post, image_data=image_data, video_filename=video_filename)


def get_post(post_id):
    with sqlite3.connect(DATABASE) as conn:
        cursor = conn.cursor()
        cursor.execute('SELECT id, title, content, image, video_filename FROM post WHERE id = ? ORDER BY id DESC', (post_id,))
        post = cursor.fetchone()
    return post
# Function to fetch all posts
def get_posts(limit=None):
    with sqlite3.connect(DATABASE) as conn:
        cursor = conn.cursor()
        if limit:
            cursor.execute('SELECT id, title, content, image, video_filename FROM post ORDER BY id DESC LIMIT ?', (limit,))
        else:
            cursor.execute('SELECT id, title, content, image, video_filename FROM post ORDER BY id DESC')
        posts = cursor.fetchall()
    return posts
def get_intro(limit=1):
    print(f"Fetching post with limit={limit}")
    with sqlite3.connect(DATABASE) as conn:
        cursor = conn.cursor()
        try:
            if limit:
                print("Fetching post with id = 864")
                cursor.execute('SELECT id, title, content, image, video_filename FROM post WHERE id = ?', (864,))
            else:
                print("Fetching all posts ordered by id DESC")
                cursor.execute('SELECT id, title, content, image, video_filename FROM post ORDER BY id DESC')
                
            posts = cursor.fetchall()
            
            return posts
        except sqlite3.OperationalError as e:
            print(f"SQLite error occurred: {e}")
            raise
# Function to fetch image data
def get_image(post_id):
    with sqlite3.connect(DATABASE) as conn:
        cursor = conn.cursor()
        cursor.execute('SELECT image FROM post WHERE id = ?', (post_id,))
        post = cursor.fetchone()
        if post and post[0]:
            return post[0]  # Return base64 encoded image data
        return None

@app.route('/home2')
def home2():
    posts = get_posts(limit=6) 
    for post in posts:
        print(post[3])# Limit to last 6 posts
    return render_template('home.html', posts=posts)

@app.route('/new', methods=['GET', 'POST'])
def new_post():
    if request.method == 'POST':
        title = request.form['title']
        content = request.form['content']
        image = request.files['image'].read() if 'image' in request.files and request.files['image'].filename != '' else None
        if image:
            image = base64.b64encode(image).decode('utf-8')
        with sqlite3.connect(DATABASE) as conn:
            cursor = conn.cursor()
            cursor.execute('INSERT INTO post (title, content, image) VALUES (?, ?, ?)', (title, content, image))
            conn.commit()
        flash('Post created successfully!', 'success')
        return redirect(url_for('home2'))
    return render_template('new_post.html')

@app.route('/edit/<int:post_id>', methods=['GET', 'POST'])
def edit_post(post_id):
    post = get_post(post_id)
    if request.method == 'POST':
        title = request.form['title']
        content = request.form['content']
        image_data = get_image(post_id)  # Get the current image data
        if 'image' in request.files and request.files['image'].filename != '':
            image = request.files['image'].read()
            image_data = base64.b64encode(image).decode('utf-8')  # Update with new image data
        with sqlite3.connect(DATABASE) as conn:
            cursor = conn.cursor()
            cursor.execute('UPDATE post SET title = ?, content = ?, image = ? WHERE id = ?', (title, content, image_data, post_id))
            conn.commit()
        flash('Post updated successfully!', 'success')
        return redirect(url_for('post', post_id=post_id))
    return render_template('edit_post.html', post=post)

@app.route('/contents')
def contents():
    posts = get_posts()
    contents_data = []
    for post in posts:
        excerpt = post[2][:300] + '...' if len(post[2]) > 300 else post[2]  # Assuming content is in the third column (index 2)
        contents_data.append({
            'id': post[0],
            'title': post[1],
            'excerpt': excerpt
        })
    return render_template('contents.html', contents_data=contents_data)

@app.route('/delete/<int:post_id>', methods=['POST'])
def delete_post(post_id):
    with sqlite3.connect(DATABASE) as conn:
        cursor = conn.cursor()
        cursor.execute('DELETE FROM post WHERE id = ?', (post_id,))
        conn.commit()
    flash('Post deleted successfully!', 'success')
    return redirect(url_for('home'))

'''
def load_txt_files(directory):
    init_db()  # Initialize the SQLite database if not already created
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    try:
        for filename in os.listdir(directory):
            if filename.endswith('.txt'):
                filepath = os.path.join(directory, filename)
                with open(filepath, 'r', encoding='utf-8') as file:
                    title = os.path.splitext(filename)[0]
                    content = file.read()
                    cursor.execute('SELECT id FROM post WHERE title = ? ORDER BY id DESC', (title,))
                    existing_post = cursor.fetchone()
                    if not existing_post:
                        cursor.execute('INSERT INTO post (title, content) VALUES (?, ?)', (title, content))
                        conn.commit()
                        logit(f'Added post: {title}')
                    else:
                        logit(f'Skipped existing post: {title}')
    except sqlite3.Error as e:
        print(f'SQLite error: {e}')
    finally:
        conn.close()
'''
@app.route('/search', methods=['GET', 'POST'])
def search():
    if request.method == 'POST':
        search_terms = request.form['search_terms']
        conn = sqlite3.connect(DATABASE)
        cursor = conn.cursor()
        
        # Define the search terms
        search_terms = search_terms.split(",")  # Split by comma to get individual search terms
        results = []
        
        # Construct the WHERE clause for the SQL query to filter rows based on all search terms
        where_conditions = []
        for term in search_terms:
            where_conditions.append(f"content LIKE ?")
        
        where_clause = " AND ".join(where_conditions)
        
        # Create a tuple of search terms with wildcard characters for the SQL query
        search_terms_tuple = tuple(f"%{term.strip()}%" for term in search_terms)
        
        # Execute the SELECT query with the constructed WHERE clause
        query = f"SELECT ROWID, title, content, image, video_filename FROM post WHERE {where_clause} ORDER BY ROWID DESC"
        rows = cursor.execute(query, search_terms_tuple)

        
        for row in rows:
            results.append((row[0], row[1], row[2], row[3], row[4]))
        
        conn.close()
        return render_template('search.html', results=results)
    
    return render_template('search.html', results=[])


def get_image_data(post_id):
    with sqlite3.connect(DATABASE) as conn:
        cursor = conn.cursor()
        cursor.execute('SELECT image FROM post WHERE id = ?', (post_id,))
        post = cursor.fetchone()
        if post and post[0]:
            return base64.b64decode(post[0])  # Decode base64 to binary
        else:
            return None

@app.route('/post/<int:post_id>', methods=['GET', 'POST'])
def show_post(post_id):
    if request.method == 'POST':
        if 'videoFile' not in request.files:
            flash('No file part')
            return redirect(request.url)
        
        file = request.files['videoFile']

        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['VUPLOAD_FOLDER'], filename))
            
            with sqlite3.connect(DATABASE) as conn:
                cursor = conn.cursor()
                cursor.execute('UPDATE post SET video_filename = ? WHERE id = ?', (filename, post_id))
                conn.commit()
                flash('Video uploaded successfully')

            return redirect(url_for('show_post', post_id=post_id))

    with sqlite3.connect(DATABASE) as conn:
        cursor = conn.cursor()
        cursor.execute('SELECT id, title, content, image, video_filename FROM post WHERE id = ? ORDER BY id DESC', (post_id,))
        post = cursor.fetchone()
        if not post:
            flash('Post not found')
            return redirect(url_for('home'))
        
        image_data = base64.b64decode(post[3]) if post[3] else None
        video_filename = post[4] if post[4] else None
    print(f"video_filename: {video_filename}")
    return render_template('post.html', post=post, image_data=image_data, video_filename=video_filename)

@app.route('/image/<int:post_id>')
def view_image(post_id):
    image_data = get_image_data(post_id)
    if image_data:
        return send_file(io.BytesIO(image_data), mimetype='image/jpeg')
    else:
        return "No image found", 404

TEXT_FILES_DIR = "static/TEXT" 
# Index route to display existing text files and create new ones
@app.route("/edit_text", methods=["GET", "POST"])
def edit_text():

    if request.method == "POST":
        filename = request.form["filename"]
        text = request.form["text"]
        save_text_to_file(filename, text)
        return redirect(url_for("edit_text"))
    else:
        # Path to the file containing list of file paths
        text_files = os.listdir(TEXT_FILES_DIR)
        text_directory='static/TEXT'
        files = sorted(text_files, key=lambda x: os.path.getmtime(os.path.join(text_directory, x)), reverse=True)
        #files=glob.glob('static/TEXT/*.txt')
        print(f'files 1: {files}')  
        # Call the function to list files by creation time
        #files = list_files_by_creation_time(files)
        print(f'files 2: {files}')
        return render_template("edit_text.html", files=files)
 # Route to edit a text file
@app.route("/edit/<filename>", methods=["GET", "POST"])
def edit(filename):
    if request.method == "POST":
        text = request.form["text"]
        save_text_to_file(filename, text)
        return redirect(url_for("index"))
    else:
        text = read_text_from_file(filename)
        return render_template("edit.html", filename=filename, text=text)
# Route to delete a text file
@app.route("/delete/<filename>")
def delete(filename):
    filepath = os.path.join(TEXT_FILES_DIR, filename)
    if os.path.exists(filepath):
        os.remove(filepath)
        print(f"File deleted: {filename}")
    return redirect(url_for("index"))


def list_files_by_creation_time(file_paths):
    """
    List files by their creation time, oldest first.
    
    Args:
    file_paths (list): List of file paths.
    
    Returns:
    list: List of file paths sorted by creation time.
    """
    # Log the start of the function
    print('Listing files by creation time...')
    
    # Create a dictionary to store file paths and their creation times
    file_creation_times = {}
    
    # Iterate through each file path
    for file_path in file_paths:
        # Get the creation time of the file
        try:
            creation_time = os.path.getctime(file_path)
            # Store the file path and its creation time in the dictionary
            file_creation_times[file_path] = creation_time
        except FileNotFoundError:
            # Log a warning if the file is not found
            print(f'File not found: {file_path}')
    
    # Sort the dictionary by creation time
    sorted_files = sorted(file_creation_times.items(), key=lambda x: x[1],reverse=True)
    
    # Extract the file paths from the sorted list
    sorted_file_paths = [file_path for file_path, _ in sorted_files]
    
    # Log the end of the function
    print('File listing complete.')
    
    # Return the sorted file paths
    return sorted_file_paths
def read_text_from_file(filename):
    filepath = os.path.join(TEXT_FILES_DIR, filename)
    with open(filepath, "r") as file:
        text = file.read()
        print(f"Text read from file: {filename}")
        return text
    

@app.route('/generate', methods=['POST'])
def generate_text():
    input_text = request.form['input_text']
    generated_text = generate_text_with_model(input_text)
    print(f"Generated text: {generated_text}")
    return jsonify({'generated_text': generated_text})

def generate_text_with_model(input_text):
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    inputs = tokenizer(input_text, return_tensors="pt")
    input_ids = inputs["input_ids"]
    
    sample_output = model.generate(
        input_ids, 
        max_length=500, 
        temperature=0.8, 
        top_p=0.9, 
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id
    )
    
    generated_text = tokenizer.decode(sample_output[0], skip_special_tokens=True)
    return generated_text

@app.route('/ask', methods=['GET', 'POST'])
def ask():
    return html_content
html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>GPT-2 Text Generation</title>
    <link rel="stylesheet" href="static/dark.css">
    <style>
        textarea {
            width: 100% !important;
            height: 60px !important;
            margin: 10px 0;
            padding: 10px;
            font-size: 16px;
        }
        input[type="submit"] {
            padding: 10px 20px;
            font-size: 16px;
            background-color: #007bff;
            color: white;
            border: none;
            cursor: pointer;
        }
        pre {
            background-color: darkgray;
            padding: 10px;
            white-space: pre-wrap;
            word-wrap: break-word;
            font-size: 22px;
        }
        #text {
            background-color: black;
            margin-top: 20px;
            font-size: 24px;
        }
    
        </style>
</head>
<body>
    <h1>GPT-2 Text Generation</h1>
    <!-- Add link home -->
    <a href="/">Home</a>
    <form id="inputForm">
        <label for="input_text">Enter Input Text:</label><br>
        <textarea class="small" id="input_text" name="input_text"></textarea><br>
        <input type="submit" value="Generate Text">
    </form>
    <pre style="color:black;" id="generated_text"></pre>
    <script>
        document.getElementById('inputForm').addEventListener('submit', async function(event) {
            event.preventDefault();
            const formData = new FormData(this);
            const response = await fetch('/generate', {
                method: 'POST',
                body: formData
            });
            const data = await response.json();
            document.getElementById('generated_text').innerHTML = '<h2>Generated Text:</h2>' + data.generated_text;
        });
    </script>
</body>
</html>
"""
def save_static_gallery_data(data):
    with open(STATIC_GALLERY_DATA_FILE, 'w') as f:
        json.dump(data, f)
    print(f'Static gallery data saved to {STATIC_GALLERY_DATA_FILE}')

def load_static_gallery_data():
    # Scan directories if static gallery data file is missing or empty
    if not os.path.exists(STATIC_GALLERY_DATA_FILE):
        print(f'{STATIC_GALLERY_DATA_FILE} not found. Scanning directories.')
        scan_directories() 
    else:           
        with open(STATIC_GALLERY_DATA_FILE, 'r') as f:
            data = json.load(f)
            print(f'Static gallery data loaded from {STATIC_GALLERY_DATA_FILE}')
            return data
    return None

# Define supported image extensions
IMAGE_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.gif', '.bmp','webp']

def scan_directories():
    image_dirs = []
    # Define the base directory to search in (static/*resources)
    base_dir = 'static'

    # Walk through directories starting from 'static'
    for root, dirs, files in os.walk(base_dir):
        # Check if the directory contains the word 'resources'
        if 'resources' in root:
            logit(f'Scanning directory: {root}')
            # Filter the files by image extensions
            image_files = [f for f in files if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS]

            # If the directory contains 20 or more images, add it to the results
            if len(image_files) >= 20:
                image_dirs.append({
                    'path': root,
                    'images': image_files
                })
                print(f'Found {len(image_files)} images in directory: {root}')

    return image_dirs
def save_scan_data(data):
    with open(SCAN_DATA_FILE, 'w') as f:
        json.dump(data, f)
    print(f'Scan data saved to {SCAN_DATA_FILE}')

def load_scan_data():
    if os.path.exists(SCAN_DATA_FILE):
        with open(SCAN_DATA_FILE, 'r') as f:
            data = json.load(f)
            print(f'Scan data loaded from {SCAN_DATA_FILE}')
            return data
    print(f'{SCAN_DATA_FILE} not found.')
    return None

def select_random_images(image_dirs):
    gallery_data = []
    for dir_data in image_dirs:
        images = dir_data['images']
        if len(images) >= 5:
            sample_images = random.sample(images, 3)
            gallery_data.append({
                'directory': dir_data['path'],
                'images': [os.path.join(dir_data['path'], img) for img in sample_images]
            })
            print(f'Selected images from directory: {dir_data["path"]}')
    return gallery_data

# get a list of images in STATIC_IMAGE_DIR
def resource_images():
    images = glob.glob(os.path.join(STATIC_IMAGE_DIR, '*.jpg'))
    images += glob.glob(os.path.join(STATIC_IMAGE_DIR, '*.png'))
    return images   
    

def copy_images_to_static(gallery_data):
    if not os.path.exists(STATIC_IMAGE_DIR):
        os.makedirs(STATIC_IMAGE_DIR)
        print(f'Created directory: {STATIC_IMAGE_DIR}')

    static_image_paths = []
    for item in gallery_data:
        static_images = []
        for img_path in item['images']:
            img_name = os.path.basename(img_path)
            dest_path = os.path.join(STATIC_IMAGE_DIR, img_name)
            shutil.copy(img_path, dest_path)
            static_images.append(dest_path)
            print(f'Copied image {img_name} to {STATIC_IMAGE_DIR}')
        static_image_paths.append({
            'directory': item['directory'],
            'images': static_images
        })
    return static_image_paths

@app.route('/gallery')
def gallery():
    scan_data = load_scan_data()
    if not scan_data:
        print('No scan data found. Scanning directories.')
        scan_data = scan_directories()
        save_scan_data(scan_data)
        gallery_data = select_random_images(scan_data)
        static_gallery_data = copy_images_to_static(gallery_data)
    else:
        static_gallery_data = load_static_gallery_data()
        if not static_gallery_data:
            print('No static gallery data found. Creating new static gallery data.')
            gallery_data = select_random_images(scan_data)
            static_gallery_data = copy_images_to_static(gallery_data)
            save_static_gallery_data(static_gallery_data)
    return render_template('gallery.html', gallery_data=static_gallery_data)


@app.route('/remove_images', methods=['GET', 'POST'])
def remove_images():
    folder = 'static/novel_images/'
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        if os.path.isfile(file_path):
        #back up images 'static/junk' with a unique uuid before removing
            shutil.copy(file_path, 'static/junk_resources/' + str(uuid.uuid4()) + filename)
            os.remove(file_path)
            print(f"Removed file: {file_path}")
    return redirect(url_for('mk_videos'))

# Directory containing the images
archived_images_dir = 'static/novel_images'
#archived_images_dir = 'static/archived-images/'  # Update this path as needed


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


#-------------------------------
@app.route('/remove_image', methods=['GET', 'POST'])
def remove_image():
    folder = 'static/archived_resources/'
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)
            print(f"Removed file: {file_path}")
    return redirect(url_for('index'))

# Directory containing the images
archived_images = 'static/archived_resources'
@app.route('/clean_archives', methods=['GET', 'POST'])
def clean_archives():
    if request.method == 'POST':
        # Get selected images
        selected_images = request.form.getlist('selected_images')
        #list of images sorted by creation time
       
        # Remove selected images
        for image in selected_images:
            image_path = os.path.join(archived_images, image)
            if os.path.exists(image_path):
                os.remove(image_path)
                print(f"Removed image: {image_path}")
        
        return redirect(url_for('clean_archives'))
    
    # Get list of images in the directory (png and jpg)
    images = [os.path.basename(img) for img in glob.glob(os.path.join(archived_images, '*.png'))]
    images += [os.path.basename(img) for img in glob.glob(os.path.join(archived_images, '*.jpg'))]
    images = sorted(images, key=lambda x: os.path.getmtime(os.path.join(archived_images, x)), reverse=True)
    print(f"clean_archives_Images: {images}")
    return render_template('clean_archives.html', images=images)

def create_backup_folder():
    backup_folder = os.path.join(os.getcwd(), "static", "Backups")
    if not os.path.exists(backup_folder):
        os.makedirs(backup_folder)
        print(f"Backup folder created at: {backup_folder}")
@app.route('/edit_files', methods=['GET', 'POST'])
def edit_files():
    filename = request.args.get('filename', '')
    directory_path = "."
    PATH = os.getcwd()
    print(f"Current Directory:, {PATH}")
    full_path = os.path.join(PATH, directory_path, filename)

    if not os.path.exists(filename):
        return "File not found", 404

    if request.method == 'POST':
        content = request.form.get('content')
        
        if content is not None:
            date_str = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
            new_filename = f"{os.path.splitext(filename)[0]}_{date_str}.txt"
            print(f'File edited and saved as: {new_filename}')

            with open(os.path.join(directory_path, new_filename), 'w') as new_file:
                new_file.write(content)

            print(f'File edited and saved as: {new_filename}')

            return send_file(os.path.join(directory_path, new_filename), as_attachment=True)

    with open(full_path, 'r') as file:
        content = file.read()

    return render_template('edit_files.html', filename=filename, content=content)

@app.route('/edit_html', methods=['POST', 'GET'])
def edit_html():
    path = TEMPLATE_DIR

    if request.method == 'POST':
        if 'load_file' in request.form:
            selected_file = request.form['selected_file']
            file_path = os.path.join(path, selected_file)

            # Load the HTML content from the file
            html_content = load_html_file(file_path)

            # Return JSON for AJAX to update the textarea
            return jsonify({'html_content': html_content})

        elif 'edited_content' in request.form:
            edited_content = request.form['edited_content']
            file_path = os.path.join(path, request.form['selected_file'])

            # Save the edited HTML content
            with open(file_path, "w") as file:
                file.write(edited_content)
                flash('File updated successfully!', 'success')

    return render_template('choose_file.html', files=choose_html())


def load_html_file(file_path):
    """Load the content of the selected HTML file."""
    with open(file_path, "r") as file:
        html_content = file.read()
    return html_content


def choose_html():
    """List all HTML files in the templates directory."""
    path = TEMPLATE_DIR
    files = glob.glob(path + "/*.html")
    # show changes first
    files = sorted(files, key=os.path.getmtime, reverse=True)
    return [os.path.basename(file) for file in files]


@app.route('/html_index')
def html_index():
    template_files = [f for f in os.listdir(TEMPLATE_DIR) if f.endswith('.html')]
    return render_template('html_index.html', template_files=template_files)


def feather_image(image, radius=50):
    """Applies a feathered transparency effect to the left and right edges of an image."""
    print(f"Applying feather effect with radius {radius} to image of size {image.size}")
    
    mask = Image.new("L", image.size, 0)
    mask.paste(255, (radius, 0, image.width - radius, image.height))
    mask = mask.filter(ImageFilter.GaussianBlur(radius))
    
    image.putalpha(mask)
    return image

def create_seamless_image(images, feather_radius=5, overlap=100):
    """Creates a seamless image by blending the provided images with feathered edges and overlap."""
    total_width = sum(img.width for img in images) - overlap * (len(images) - 1)
    max_height = max(img.height for img in images)

    print(f"Creating combined image of size {total_width}x{max_height}")
    
    combined_image = Image.new("RGBA", (total_width, max_height))

    x_offset = 0
    for i, img in enumerate(images):
        feathered_img = feather_image(img, feather_radius)
        combined_image.paste(feathered_img, (x_offset, 0), feathered_img)
        x_offset += img.width - overlap
        print(f"Image {i+1} pasted at position {x_offset}")

    return combined_image

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

@app.route('/create_video', methods=['POST', 'GET'])
def create_video():
    """Endpoint to create a scrolling video from a set of images."""
    #copy static/temp_exp/TEMP2.mp4 static/temp_exp/diagonal1.mp4
    #shutil.copy("static/temp_exp/TEMP2.mp4", "static/temp_exp/diagonal1.mp4")
    try:
        vid_directory = 'static/novel_images'

        # Get all image files in the directory
        #image_files = glob.glob(os.path.join(vid_directory, '*.png'))
        image_files = glob.glob(os.path.join(vid_directory, '*.png')) + glob.glob(os.path.join(vid_directory, '*.jpg'))

        if not image_files:
            print("No image files found.")
            return jsonify({"error": "No image files found."}), 404

        # Sort files by modification time
        image_files = sorted(image_files, key=os.path.getmtime, reverse=True)

        if len(image_files) < 6:
            print("Less than 6 images found. Adjusting the number of selected images.")

        images = [Image.open(img).convert('RGBA').resize((512, 768), resample=Image.LANCZOS) for img in image_files]

        # Create the seamless image
        seamless_image_path = 'static/seamless_image.png'
        seamless_image = create_seamless_image(images, feather_radius=10, overlap=100)
        seamless_image.save(seamless_image_path)
        print(f"Seamless image saved as {seamless_image_path}")

        # Create the scrolling video
        video_path = 'static/seamless_video.mp4'
        video_duration = 34
        video_size = (512, 768)
        make_scrolling_video(seamless_image_path, video_path, video_duration, video_size)
        
        # Optionally run external process if needed
        add_title_image(video_path, hex_color="#A52A2A")
        return redirect(url_for('index'))
    except Exception as e:
        return jsonify({"error": str(e)}), 500
        
def createvideo():
    try:
        vid_directory = 'static/novel_images'
        # Get all image files in the directory
        #image_files = glob.glob(os.path.join(vid_directory, '*.png'))
        image_files = glob.glob(os.path.join(vid_directory, '*.png')) + glob.glob(os.path.join(vid_directory, '*.jpg'))

        if not image_files:
            print("No image files found.")
            return jsonify({"error": "No image files found."}), 404

        # Sort files by modification time
        image_files = sorted(image_files, key=os.path.getmtime, reverse=True)

        if len(image_files) < 6:
            print("Less than 6 images found. Adjusting the number of selected images.")

        images = [Image.open(img).convert('RGBA').resize((512, 768), resample=Image.LANCZOS) for img in image_files[:10]]

        # Create the seamless image
        seamless_image_path = 'static/seamless_image.png'
        seamless_image = create_seamless_image(images, feather_radius=5, overlap=50)
        seamless_image.save(seamless_image_path)
        print(f"Seamless image saved as {seamless_image_path}")

        # Create the scrolling video
        video_path = 'static/seamless_videoX.mp4'
        video_duration = 56
        video_size = (512, 768)
        make_scrolling_video(seamless_image_path, video_path, video_duration, video_size)
        
        # Optionally run external process if needed
        add_title_image(video_path, hex_color="#A52A2A")
        return redirect(url_for('index'))
    except Exception as e:
        print(f"Error creating video 1938: {e}")
        return jsonify({"error": str(e)}), 500


def add_title_image(video_path, hex_color = "#A52A2A"):
    hex_color=random.choice(["#A52A2A","#ad1f1f","#16765c","#7a4111","#9b1050","#8e215d","#2656ca"])
    # Define the directory path
    directory_path = "temp"
    # Check if the directory exists
    if not os.path.exists(directory_path):
        # If not, create it
        os.makedirs(directory_path)
        print(f"Directory '{directory_path}' created.")
    else:
        print(f"Directory '{directory_path}' already exists.") 
    # Load the video file and title image
    video_clip = VideoFileClip(video_path)
    print(video_clip.size)
    # how do i get the width and height of the video
    width, height = video_clip.size
    get_duration = video_clip.duration
    print(get_duration, width, height)
    title_image_path = "static/assets/512x568_border.png"
    # Set the desired size of the padded video (e.g., video width + padding, video height + padding)
    padded_size = (width + 50, height + 50)

    # Calculate the position for centering the video within the larger frame
    x_position = (padded_size[0] - video_clip.size[0]) / 2
    y_position = (padded_size[1] - video_clip.size[1]) / 2
    #hex_color = "#09723c"
    # Remove the '#' and split the hex code into R, G, and B components
    r = int(hex_color[1:3], 16)
    g = int(hex_color[3:5], 16)
    b = int(hex_color[5:7], 16)

    # Create an RGB tuple
    rgb_tuple = (r, g, b)

    # Create a blue ColorClip as the background
    blue_background = ColorClip(padded_size, color=rgb_tuple)

    # Add the video clip on top of the red background
    padded_video_clip = CompositeVideoClip([blue_background, video_clip.set_position((x_position, y_position))])
    padded_video_clip = padded_video_clip.set_duration(video_clip.duration)
    #title_image_path = "/home/jack/Desktop/EXPER/static/assets/Title_Image02.png"
    # Load the title image
    title_image = ImageClip(title_image_path)

    # Set the duration of the title image
    title_duration = video_clip.duration
    title_image = title_image.set_duration(title_duration)

    print(video_clip.size)
    # Position the title image at the center and resize it to fit the video dimensions
    #title_image = title_image.set_position(("left", "top"))
    title_image = title_image.set_position((0, -5))
    #video_clip.size = (620,620)
    title_image = title_image.resize(padded_video_clip.size)

    # Position the title image at the center and resize it to fit the video dimensions
    #title_image = title_image.set_position(("center", "center")).resize(video_clip.size)

    # Create a composite video clip with the title image overlay
    composite_clip = CompositeVideoClip([padded_video_clip, title_image])
    # Limit the length to video duration
    composite_clip = composite_clip.set_duration(video_clip.duration)
    # Load a random background music
    mp3_files = glob.glob("/mnt/HDD500/collections/music_dark/*.mp3")
    random.shuffle(mp3_files)

    # Now choose a random MP3 file from the shuffled list
    mp_music = random.choice(mp3_files)
    get_duration = AudioFileClip(mp_music).duration
    # Load the background music without setting duration
    music_clip = AudioFileClip(mp_music)
    # Fade in and out the background music
    #music duration is same as video
    music_clip = music_clip.set_duration(video_clip.duration)
    # Fade in and out the background music
    fade_duration = 1.0
    music_clip = music_clip.audio_fadein(fade_duration).audio_fadeout(fade_duration)
    # Set the audio of the composite clip to the background music
    composite_clip = composite_clip.set_audio(music_clip)
    uid = uuid.uuid4().hex
    output_path = 'static/temp_exp/TEMP2X.mp4'
    # Export the final video with the background music
    composite_clip.write_videofile(output_path)
    mp4_file =  f"/mnt/HDD500/collections/vids/Ready_Post_{uid}.mp4"
    shutil.copyfile(output_path, mp4_file)     
    print(mp4_file)
    VIDEO = output_path
    return VIDEO
def add_title(video_path, hex_color = "#A52A2A"):
    hex_color=random.choice(["#A52A2A","#ad1f1f","#16765c","#7a4111","#9b1050","#8e215d","#2656ca"])
    # Define the directory path
    directory_path = "tempp"
    # Check if the directory exists
    if not os.path.exists(directory_path):
        # If not, create it
        os.makedirs(directory_path)
        print(f"Directory '{directory_path}' created.")
    else:
        print(f"Directory '{directory_path}' already exists.") 
    # Load the video file and title image
    video_clip = VideoFileClip(video_path)
    print(video_clip.size)
    # how do i get the width and height of the video
    width, height = video_clip.size
    get_duration = video_clip.duration
    print(get_duration, width, height)
    title_image_path = "static/assets/512x568_border.png"
    # Set the desired size of the padded video (e.g., video width + padding, video height + padding)
    padded_size = (width + 50, height + 50)

    # Calculate the position for centering the video within the larger frame
    x_position = (padded_size[0] - video_clip.size[0]) / 2
    y_position = (padded_size[1] - video_clip.size[1]) / 2
    #hex_color = "#09723c"
    # Remove the '#' and split the hex code into R, G, and B components
    r = int(hex_color[1:3], 16)
    g = int(hex_color[3:5], 16)
    b = int(hex_color[5:7], 16)

    # Create an RGB tuple
    rgb_tuple = (r, g, b)

    # Create a blue ColorClip as the background
    blue_background = ColorClip(padded_size, color=rgb_tuple)

    # Add the video clip on top of the red background
    padded_video_clip = CompositeVideoClip([blue_background, video_clip.set_position((x_position, y_position))])
    padded_video_clip = padded_video_clip.set_duration(video_clip.duration)
    #title_image_path = "/home/jack/Desktop/EXPER/static/assets/Title_Image02.png"
    # Load the title image
    title_image = ImageClip(title_image_path)

    # Set the duration of the title image
    title_duration = video_clip.duration
    title_image = title_image.set_duration(title_duration)

    print(video_clip.size)
    # Position the title image at the center and resize it to fit the video dimensions
    #title_image = title_image.set_position(("left", "top"))
    title_image = title_image.set_position((0, -5))
    #video_clip.size = (620,620)
    title_image = title_image.resize(padded_video_clip.size)

    # Position the title image at the center and resize it to fit the video dimensions
    #title_image = title_image.set_position(("center", "center")).resize(video_clip.size)

    # Create a composite video clip with the title image overlay
    composite_clip = CompositeVideoClip([padded_video_clip, title_image])
    # Limit the length to video duration
    composite_clip = composite_clip.set_duration(video_clip.duration)
    # Load a random background music
    mp3_files = glob.glob("/mnt/HDD500/collections/music_dark/*.mp3")
    random.shuffle(mp3_files)

    # Now choose a random MP3 file from the shuffled list
    mp_music = random.choice(mp3_files)
    get_duration = AudioFileClip(mp_music).duration
    # Load the background music without setting duration
    music_clip = AudioFileClip(mp_music)
    # Fade in and out the background music
    #music duration is same as video
    music_clip = music_clip.set_duration(video_clip.duration)
    # Fade in and out the background music
    fade_duration = 1.0
    music_clip = music_clip.audio_fadein(fade_duration).audio_fadeout(fade_duration)
    # Set the audio of the composite clip to the background music
    composite_clip = composite_clip.set_audio(music_clip)
    uid = uuid.uuid4().hex
    output_path = 'static/temp_exp/TEMP1X.mp4'
    # Export the final video with the background music
    composite_clip.write_videofile(output_path)
    mp4_file =  f"/mnt/HDD500/collections/vids/Ready_Post_{uid}.mp4"
    shutil.copyfile(output_path, mp4_file)     
    print(mp4_file)
    VIDEO = output_path
    return VIDEO
def ensure_dir_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def next_points(point, imgsize, avoid_points=[], shuffle=True):
    point_list = [p for p in 
                  [(point[0], point[1]+1), (point[0], point[1]-1), 
                   (point[0]+1, point[1]), (point[0]-1, point[1])]
                  if 0 < p[0] < imgsize[0]//2 and 0 < p[1] < imgsize[1] and p not in avoid_points]

    if shuffle:
        random.shuffle(point_list)

    return point_list

def degrade_color(color, degradation=10):
    return tuple(min(c + degradation, 255) for c in color)

def spread(img, point, color, max_white=100, degradation=10):
    if color[0] <= max_white and img.getpixel(point)[0] > color[0]:
        img.putpixel(point, color)
        points = next_points(point, img.size, shuffle=False)
        color = degrade_color(color, degradation)
        for point in points:
            spread(img, point, color)

def binarize_array(numpy_array, threshold=200):
    return np.where(numpy_array > threshold, 255, 0)

def processr_image(seed_count, seed_max_size, imgsize=(510, 766), count=0):
    margin_h, margin_v = 60, 60
    color = (0, 0, 0)
    img = Image.new("RGB", imgsize, "white")
    old_points = []
    posible_root_points = []

    for seed in range(seed_count):
        point = None
        while not point or point in old_points:
            point = (random.randrange(0 + margin_h, imgsize[0]//2), 
                     random.randrange(0 + margin_v, imgsize[1] - margin_v))
        old_points.append(point)
        posible_root_points.append(point)
        img.putpixel(point, color)

        seedsize = random.randrange(0, seed_max_size)
        flow = 0
        for progress in range(seedsize):
            flow += 1
            points = next_points(point, imgsize, old_points)
            try:
                point = points.pop()
            except IndexError:
                posible_root_points.remove(point)
                for idx in reversed(range(len(posible_root_points))):
                    points = next_points(posible_root_points[idx], imgsize, old_points)
                    try:
                        point = points.pop()
                        flow = 0
                        break
                    except IndexError:
                        posible_root_points.pop()
                if not point:
                    break

            old_points.append(point)
            posible_root_points.append(point)
            img.putpixel(point, color)

            for surr_point in points:
                spread(img, surr_point, degrade_color(color))

    cropped = img.crop((0, 0, imgsize[0]//2, imgsize[1]))
    img = img.transpose(Image.FLIP_LEFT_RIGHT)
    img.paste(cropped, (0, 0, imgsize[0]//2, imgsize[1]))
    img = img.filter(ImageFilter.GaussianBlur(radius=10))
    
    filename0 = "static/images/blot.png"
    img.save(filename0)

    im_grey = img.convert('LA')
    mean = np.mean(np.array(im_grey)[:, :, 0])

    image_file = Image.open(filename0)
    imagex = image_file.convert('L')
    imagey = np.array(imagex)
    imagez = binarize_array(imagey, mean)

    temp_filename = "static/images/tmmmp.png"
    cv2.imwrite(temp_filename, imagez)

    final_filename = time.strftime("static/archived-images/GOODblots%Y%m%d%H%M%S.png")
    ImageOps.expand(Image.open(temp_filename).convert("L"), border=1, fill='white').save(final_filename)

    print("GoodBlot: ", count)
    return final_filename

    def weight_boundary(graph, src, dst, n):
        default = {'weight': 0.0, 'count': 0}
        count_src = graph[src].get(n, default)['count']
        count_dst = graph[dst].get(n, default)['count']
        weight_src = graph[src].get(n, default)['weight']
        weight_dst = graph[dst].get(n, default)['weight']
        count = count_src + count_dst
        return {
            'count': count,
            'weight': (count_src * weight_src + count_dst * weight_dst) / count
        }

    def merge_boundary(graph, src, dst):
        pass

    labels2 = future.graph.merge_hierarchical(labels, g, thresh=0.08, rag_copy=False,
                                              in_place_merge=True,
                                              merge_func=merge_boundary,
                                              weight_func=weight_boundary)

    out = color.label2rgb(labels2, img, kind='avg')
    
    # Save the processed image
    output_filename = os.path.join(app.config['UPLOAD_FOLDER'], f"processed_{time.time()}.png")
    imsave(output_filename, out)
    return output_filename

# Route for the main page
@app.route('/uploadfile', methods=['GET', 'POST'])
def uploadfile():
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Process the image and get the output path
            processed_image_path = process_image(filepath)
            
            return redirect(url_for('uploaded_file', filename=os.path.basename(processed_image_path)))
    return '''
    <!doctype html>
    <title>Upload an image</title>
    <h1>Upload an image for segmentation processing</h1>
    <form method=post enctype=multipart/form-data>
      <input type=file name=file>
      <input type=submit value=Upload>
    </form>
    '''
    
try:
    os.makedirs("static/outlines")
except FileExistsError:
    # directory already exists
    pass
def auto_canny(image, sigma=0.33):
    # compute the median of the single channel pixel intensities
    v = np.median(image)
    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)
    # return the edged image
    return edged

#image = cv2.imread('mahotastest/orig-color.png')
def change_extension(orig_file,new_extension):
    p = change_ext(orig_file)
    new_name = p.rename(p.with_suffix(new_extension))
    return new_name
    
def FilenameByTime(directory):
    timestamp = str(time.time()).replace('.', '')
    filename = f"{directory}/{timestamp}.png"
    return filename

def auto_canny(image, sigma=0.33):
    v = np.median(image)
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)
    return edged

import os
import cv2
from PIL import Image
import shutil

def outlineJ(filename, sigma=0.33):
    # Load the image and apply Canny edge detection
    image = cv2.imread(filename)
    edged = auto_canny(image, sigma=sigma)
    
    # Invert the colors for black-and-white outlines
    inverted = cv2.bitwise_not(edged)
    
    # Paths to save the images
    temp_path = "static/outlines/temp2.png"
    outline_path = "static/outlines/outlined.png"
    transparent_path = "static/outlines/transparent_outline.png"

    # Save the inverted black-and-white image
    cv2.imwrite(temp_path, inverted)

    # Open the black-and-white outline image
    frontimage = Image.open(temp_path).convert("RGBA")  # Load as RGBA for transparency

    # Process to create the black outline with transparent background
    datas = frontimage.getdata()

    newData = []
    for item in datas:
        # If the pixel is white, make it transparent
        if item[0] > 200 and item[1] > 200 and item[2] > 200:  # Adjust as necessary
            newData.append((255, 255, 255, 0))
        else:
            newData.append((0, 0, 0, 255))  # Keep black as is

    frontimage.putdata(newData)
    frontimage.save(transparent_path)  # Save the transparent outline

    # Open the original image to apply the outline
    background = Image.open(filename).convert("RGBA")
    background.paste(frontimage, (3, 3), frontimage)  # Paste with transparency
    
    # Save the outlined image
    background.save(outline_path)

    # Save the outline on a white background
    outline_on_white = Image.new("RGBA", background.size, "WHITE")
    outline_on_white.paste(frontimage, (0, 0), frontimage)
    outline_on_white.save(temp_path)

    # Save the images with timestamps
    unique_id = uuid.uuid4().hex
    #savefile = f"static/outlines/{unique_id}.png"
    
    savefile = FilenameByTime("static/outlines")
    shutil.copyfile(transparent_path, f"static/archived-images/{unique_id}outlines_transparent.png")
    shutil.copyfile(outline_path, f"static/archived-images/{unique_id}_outlined.png")
    shutil.copyfile(temp_path, f"static/archived-images/{unique_id}_bw.png")
    
    return outline_path, transparent_path, temp_path


@app.route('/outlinefile', methods=['GET', 'POST'])
def outlinefile():
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Process the image and get the output path
            savefile, display_filename, temp_filename = outlineJ(filepath, sigma=0.33)
            return render_template('outlines.html', filename=display_filename, temp_filename=temp_filename)
    return '''
    <!doctype html>
    <title>Upload an image</title>
    <h1>Upload an image for outline processing</h1>
    <form method=post enctype=multipart/form-data>
      <input type=file name=file>
      <input type=submit value=Upload>
    </form>
    '''

#def allowed_file(filename):
#    return '.' in filename and filename.rsplit('.', 1)[1].lower() in{'png', 'jpg', 'jpeg', 'gif'}

# SQLite database setup functions
def get_db_connection():
    """
    Establishes a connection to the SQLite database.
    """
    try:
        conn = sqlite3.connect(DATABASEF)
        conn.row_factory = sqlite3.Row
        print("Database connection established.")
        return conn
    except sqlite3.Error as e:
        print(f"Error establishing database connection: {e}")
        traceback.print_exc()
        return None

def create_db():
    """
    Initializes the database by creating necessary tables if they don't exist.
    """
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS functions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                function_text TEXT NOT NULL
            )
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS metadata (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                key TEXT NOT NULL,
                value TEXT NOT NULL
            )
        ''')
        conn.commit()
        conn.close()
        print(f"Database {DATABASEF} initialized.")
    except sqlite3.Error as e:
        print(f"Error initializing database: {e}")
        traceback.print_exc()

def read_functions():
    """
    Reads all function texts from the database.
    """
    print("Reading functions from database...")
    try:
        conn = get_db_connection()
        if conn is None:
            print("Failed to establish database connection.")
            return []
        cursor = conn.cursor()
        cursor.execute('SELECT function_text FROM functions')
        functions = [row[0] for row in cursor.fetchall()]
        conn.close()
        print("Functions retrieved from database.")
        return functions
    except sqlite3.Error as e:
        print(f"Error reading functions: {e}")
        traceback.print_exc()
        return []

def insert_function(function_text):
    """
    Inserts a new function text into the database.
    """
    try:
        print("Inserting function into database...")
        conn = get_db_connection()
        if conn is None:
            print("Failed to establish database connection.")
            return
        cursor = conn.cursor()
        cursor.execute('INSERT INTO functions (function_text) VALUES (?)', (function_text,))
        conn.commit()
        conn.close()
        print("Function inserted into database.")
    except sqlite3.Error as e:
        print(f"Error inserting function: {e}")
        traceback.print_exc()

def insert_functions():
    """
    Inserts functions from 'con_html.txt' into the database if not already initialized.
    """
    print("Checking if functions need to be inserted into the database...")
    try:
        conn = get_db_connection()
        if conn is None:
            print("Failed to establish database connection.")
            return
        cursor = conn.cursor()
        cursor.execute("SELECT value FROM metadata WHERE key='initialized'")
        result = cursor.fetchone()
        if result is None:
            print("Initializing and inserting functions from 'con_html.txt'...")
            with open('all_html.txt', 'r', encoding='utf-8', errors='ignore') as file:
                content = file.read()
            # Assuming functions are separated by '\n.\n'
            segments = content.strip().split('@app')
            for segment in segments:
                cleaned_segment = segment.strip()
                if cleaned_segment:
                    cursor.execute('INSERT INTO functions (function_text) VALUES (?)', (cleaned_segment,))
            cursor.execute("INSERT INTO metadata (key, value) VALUES ('initialized', 'true')")
            conn.commit()
            print("Functions inserted into database.")
        else:
            print("Functions already inserted into database.")
        conn.close()
    except sqlite3.Error as e:
        print(f"Error inserting functions into database: {e}")
        traceback.print_exc()

def get_last_function():
    """
    Retrieves the most recently inserted function from the database.
    """
    print("Retrieving the last function from the database...")
    try:
        conn = get_db_connection()
        if conn is None:
            print("Failed to establish database connection.")
            return None
        cursor = conn.cursor()
        cursor.execute('SELECT function_text FROM functions ORDER BY id DESC LIMIT 1')
        result = cursor.fetchone()
        conn.close()
        if result:
            print("Last function retrieved successfully.")
            return result[0]
        else:
            print("No functions found in the database.")
            return None
    except sqlite3.Error as e:
        print(f"Error retrieving last function: {e}")
        traceback.print_exc()
        return None

@app.route('/index_code')
def index_code():
    """
    Renders the main index page with the latest function.
    """
    functions = get_last_function()
    return render_template('index_code.html', functions=functions)

@app.route('/save', methods=['POST'])
def save():
    """
    Saves the provided code and generates suggestions.
    """
    code = request.form['code']
    suggestions = generate_suggestions(code)
    return {'suggestions': suggestions}

def generate_suggestions(code):
    """
    Generates suggestions based on the last two words of the provided code.
    Each suggestion is approximately 400 characters long.
    """
    print("Generating suggestions...")
    functions = read_functions()

    if not functions:
        print("No functions available to generate suggestions.")
        return []

    # Retrieve the last line from the code
    lines = code.strip().split('\n')
    last_line = lines[-1] if lines else ''
    print(f"Last line of code: '{last_line}'")

    # Split the last line into words and get the last two words
    words = last_line.split()
    last_two_words = ' '.join(words[-2:]) if len(words) >=2 else last_line
    print(f"Last two words: '{last_two_words}'")

    # Function to split snippet based on last_two_words and return completion
    def split_snippet(snippet, last_two_words):
        index = snippet.rfind(last_two_words)
        if index != -1:
            completion = snippet[index + len(last_two_words):].strip()
            return completion
        return snippet.strip()

    # Search for matching snippets based on the last two words
    matching_snippets = []
    found_indices = set()  # To store indices of found snippets to avoid duplicates

    for i, snippet in enumerate(functions, start=1):
        if last_two_words in snippet:
            if i not in found_indices:
                found_indices.add(i)
                completion = split_snippet(snippet, last_two_words)
                formatted_snippet = f"<pre>{i}: {completion}</pre>"
                # Adjust the snippet length to approximately 400 characters
                if len(formatted_snippet) > 400:
                    formatted_snippet = formatted_snippet[:397] + '...'
                matching_snippets.append(formatted_snippet)
                print(f"Added snippet {i}: {formatted_snippet}")

    # Return up to 20 suggestions, limited to 5 for demonstration purposes
    suggestions = matching_snippets[:5]
    print(f"Generated {len(suggestions)} suggestions.")
    return suggestions

@app.route('/save_code', methods=['POST'])
def save_code():
    """
    Saves the provided code to the database.
    """
    code = request.data.decode('utf-8')
    print(f"Received code to save: {code[:50]}...")  # Log first 50 characters for brevity
    if code:
        insert_function(code)
        return 'Code saved successfully', 200
    else:
        print("No code provided in the request.")
        return 'No code provided in the request', 400

@app.route('/functions', methods=['GET', 'POST'])
def get_functions():
    """
    Retrieves all functions from the database and returns them as JSON.
    """
    print("Fetching all functions from the database.")
    conn = get_db_connection()
    if conn is None:
        print("Failed to establish database connection.")
        return jsonify([]), 500
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM functions')
    functions = cursor.fetchall()
    conn.close()
    print(f"Retrieved {len(functions)} functions from the database.")
    return jsonify([dict(ix) for ix in functions])

@app.route('/functions/<int:id>', methods=['PUT', 'GET'])
def update_response(id):
    """
    Updates the function text for a given function ID.
    """
    new_response = request.json.get('function_text')
    print(f"Updating function ID {id} with new text.")
    if not new_response:
        print("No new function text provided.")
        return jsonify({'status': 'failure', 'message': 'No function text provided'}), 400
    try:
        conn = get_db_connection()
        if conn is None:
            print("Failed to establish database connection.")
            return jsonify({'status': 'failure', 'message': 'Database connection failed'}), 500
        cursor = conn.cursor()
        cursor.execute('UPDATE functions SET function_text = ? WHERE id = ?', (new_response, id))
        conn.commit()
        conn.close()
        print(f"Function ID {id} updated successfully.")
        return jsonify({'status': 'success', 'message': 'Function updated successfully'})
    except sqlite3.Error as e:
        print(f"Error updating function ID {id}: {e}")
        traceback.print_exc()
        return jsonify({'status': 'failure', 'message': 'Error updating function'}), 500

@app.route('/view_functions', methods=['GET', 'POST'])
def view_functions():
    """
    Renders a page to view all functions.
    """
    print("Rendering view_functions page.")
    conn = get_db_connection()
    if conn is None:
        print("Failed to establish database connection.")
        return render_template('view_functions.html', data=[])
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM functions')
    data = cursor.fetchall()
    conn.close()
    print(f"Retrieved {len(data)} functions for viewing.")
    return render_template('view_functions.html', data=data)

@app.route('/update_function', methods=['POST', 'GET'])
def update_function():
    """
    Updates a specific function based on form data and redirects to the view page.
    """
    id = request.form.get('id')
    new_function_text = request.form.get('function_text')
    print(f"Received update for function ID {id}.")
    if not id or not new_function_text:
        print("Missing function ID or new function text in the request.")
        return redirect(url_for('view_functions'))
    try:
        conn = get_db_connection()
        if conn is None:
            print("Failed to establish database connection.")
            return redirect(url_for('view_functions'))
        cursor = conn.cursor()
        cursor.execute('UPDATE functions SET function_text = ? WHERE id = ?', (new_function_text, id))
        conn.commit()
        conn.close()
        print(f"Function ID {id} updated successfully via form.")
        return redirect(url_for('view_functions'))
    except sqlite3.Error as e:
        print(f"Error updating function ID {id}: {e}")
        traceback.print_exc()
        return redirect(url_for('view_functions'))

def get_suggestions(search_term):
    try:
        # Create a database connection
        conn = sqlite3.connect(':memory:')
        c = conn.cursor()

        # Create table
        c.execute('''CREATE TABLE IF NOT EXISTS dialogue
                     (id INTEGER PRIMARY KEY,
                      search_term TEXT,
                      ChatGPT_PAIR TEXT,
                      ChatGPT_PAIRb BLOB
                      )''')
        conn.commit()

        # Perform operations
        cnt = 0
        DATA = set()
        INDEX = '----SplitHere------'
        with open("app.py", "r") as data:
            Lines = data.read()
            lines = Lines.replace(search_term, INDEX + search_term)
            lines = lines.split(INDEX)
            for line in lines:
                if search_term in line:
                    cnt += 1
                    DATA.add(f'{line[:1200]}')
                    # Insert dialogue pair into the table
                    c.execute("INSERT INTO dialogue (search_term, ChatGPT_PAIR, ChatGPT_PAIRb) VALUES (?, ?, ?)",
                              (search_term, line, line.encode('utf-8')))
                    conn.commit()

        # Close the database connection
        conn.close()
        return DATA
    except Exception as e:
        print(f"An error occurred: {e}")
        return set()

@app.route('/search_file', methods=['GET', 'POST'])
def search_file():
    search_term = request.args.get('q', '') if request.method == 'GET' else request.form.get('q', '')
    if search_term:
        data = get_suggestions(search_term)
        return render_template_string('''
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Search Results</title>
<link rel="stylesheet" href="static/dark.css">
    <style>
        textarea {
            width: 100% !important;
            height: 60px !important;
            margin: 10px 0;
            padding: 10px;
            font-size: 16px;
        }
        input[type="submit"] {
            padding: 10px 20px;
            font-size: 16px;
            background-color: #007bff;
            color: white;
            border: none;
            cursor: pointer;
        }
        pre {
            background-color: darkgray;
            padding: 10px;
            white-space: pre-wrap;
            word-wrap: break-word;
            font-size: 22px;
        }
        #text {
            background-color: black;
            margin-top: 20px;
            font-size: 24px;
        }
    
        </style>
        </head>
        <body>
        <a style="color:navy;style="font-size:4vw;" href="/search_file">Search Again</a><br>
        <a style="color:navy;style="font-size:4vw;" href="/">Home</a>
            <h1 style="font-size:4vw;>Search Results for "{{ search_term }}"</h1>
            {% for item in data %}
                <div style="border-bottom: 1px solid #ccc; padding: 10px;">
                    <pre style="color:navy;font-size:24px;">{{ item }}</pre>
                </div>
            {% endfor %}
            
        </body>
        </html>
        ''', search_term=search_term, data=data)
    else:
        return render_template_string('''
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Enter Search Term</title>
        </head>
        <body>
            <h1 style="font-size:4vw;">Enter a Search Term</h1>
            <form action="/search_file" method="post">
                <input style="font-size:4vw;" type="text" name="q" placeholder="Enter search term">
                <input style="font-size:4vw;" type="submit" value="Search">
            </form>
        </body>
        </html>
        ''')
@app.route('/convert-images')
def convert_images_route():
    try:
        convert_images()
        return redirect(url_for('mk_videos'))
    except Exception as e:
        return jsonify(error=str(e)), 500

@app.route('/create-video')
def create_video_route():
    try:
        createvideo()
        return redirect(url_for('mk_videos'))
    except Exception as e:
        return jsonify(error=str(e)), 500

@app.route('/refresh-video')
def refresh_video_route():
    try:
        subprocess.run(['/home/jack/miniconda3/envs/cloned_base/bin/python', 'refresh_video.py'], check=True)
        return redirect(url_for('mk_videos'))
    except subprocess.CalledProcessError as e:
        return jsonify(error=str(e)), 500

@app.route('/best-flipbook')
def best_flipbook_route():
    try:
        subprocess.run(['/home/jack/miniconda3/envs/cloned_base/bin/python', 'Best_FlipBook'], check=True)
        return redirect(url_for('mk_videos'))
    except subprocess.CalledProcessError as e:
        return jsonify(error=str(e)), 500


@app.route('/diagonal-transition')
def diagonal_transition_route():
    try:
        subprocess.run(['/home/jack/miniconda3/envs/cloned_base/bin/python', 'diagonal_transition'], check=True)
        video_path = 'static/temp_exp/diagonal1.mp4'
        add_title(video_path, hex_color="#A52A2A")
        return redirect(url_for('mk_videos'))
    
    except subprocess.CalledProcessError as e:
        return jsonify(error=str(e)), 500

@app.route('/slide')
def slide_route():
    try:
        subprocess.run(['/bin/bash', 'slide'], check=True)
        return redirect(url_for('mk_videos'))
    except subprocess.CalledProcessError as e:
        return jsonify(error=str(e)), 500

@app.route('/zoomx4')
def zoomx4_route():
    try:
        subprocess.run(['/bin/bash', 'zoomX4'], check=True)
        return redirect(url_for('mk_videos'))
    except subprocess.CalledProcessError as e:
        return jsonify(error=str(e)), 500
    
@app.route('/zoomy4')
def zoomy4_route():
    try:
        subprocess.run(['/bin/bash', 'zoomY4'], check=True)
        return redirect(url_for('mk_videos'))
    except subprocess.CalledProcessError as e:
        return jsonify(error=str(e)), 500

@app.route('/vertical-scroll')
def vertical_scroll_route():
    try:
        subprocess.run(['/home/jack/miniconda3/envs/cloned_base/bin/python', 'vertical_scroll'], check=True)
        return redirect(url_for('mk_videos'))
    except subprocess.CalledProcessError as e:
        return jsonify(error=str(e)), 500

@app.route('/add-title')
def add_title_route():
    try:
        video_path = 'static/temp_exp/diagonal1.mp4'
        add_title(video_path, hex_color="#A52A2A")
        return redirect(url_for('mk_videos'))
    except Exception as e:
        return jsonify(error=str(e)), 500

@app.route('/join-video')
def join_video_route():
    try:
        subprocess.run(['/home/jack/miniconda3/envs/cloned_base/bin/python', 'join_all_videos'], check=True)
        return redirect(url_for('mk_videos'))
    except subprocess.CalledProcessError as e:
        return jsonify(error=str(e)), 500

@app.route('/refresh-all')
def refresh_all_route():
    try:
        # Call each route sequentially
        convert_and_resize_images_route()
        create_video_route()
        refresh_video_route()
        best_flipbook_route()
        diagonal_transition_route()
        #blendem_route()
        slide_route()
        zoomx4_route()
        vertical_scroll_route()
        add_title_route()
        join_video_route()
        resize_mp4_route

        return redirect(url_for('create_video'))
    except Exception as e:
        return jsonify(error=str(e)), 500

@app.route('/convert_and_resize_images')
def convert_and_resize_images_route():
    img_dir = 'static/novel_images'

    # Ensure the directory exists
    if not os.path.exists(img_dir):
        return "Directory not found", 404

    # Get all files in the directory and sort by creation time
    image_files = [f for f in os.listdir(img_dir) if f.lower().endswith(('.png', '.jpg'))]
    image_files = sorted(image_files, key=lambda x: os.path.getctime(os.path.join(img_dir, x)), reverse=True)

    

    # Iterate over sorted images
    for file in image_files:
        img_file_path = os.path.join(img_dir, file)

        # Convert to JPG if PNG and set the new path
        if file.lower().endswith('.png'):
            new_file_path = os.path.splitext(img_file_path)[0] + '.jpg'
        else:
            new_file_path = img_file_path

        try:
            # Open the image
            with Image.open(img_file_path) as img:
                # Resize image to height of 768 while maintaining aspect ratio
                width, height = img.size
                new_height = 768
                new_width = int((new_height / height) * width)
                resized_img = img.resize((new_width, new_height))

                # Center crop to 512x768
                left = (new_width - 512) / 2
                top = 0
                right = (new_width + 512) / 2
                bottom = 768
                cropped_img = resized_img.crop((left, top, right, bottom))

                # Convert to RGB (only for PNGs)
                if file.lower().endswith('.png'):
                    cropped_img = cropped_img.convert('RGB')

                # Save the resulting image as JPG
                cropped_img.save(new_file_path, 'JPEG')

        except Exception as e:
            continue

    return redirect(url_for('conversion_complete'))
@app.route('/conversion_complete')
def conversion_complete():
    return "All PNG and JPG files have been converted, resized to 512x768, and saved."

@app.route('/base')
def base():
    return render_template('base_1.html')

# Step 1: Resize videos and save as basename_512x768.mp4
def resize_videos(directory, target_size=(512, 768)):
    for filename in os.listdir(directory):
        if filename.endswith("X.mp4") and not filename.endswith("_512x768.mp4"):
            filepath = os.path.join(directory, filename)

            # Load the video file
            video_clip = VideoFileClip(filepath)

            # Resize the video to the target size (512x768)
            resized_clip = video_clip.resize(newsize=target_size)

            # Save the resized video with the new name basename_512x768.mp4
            new_filename = os.path.splitext(filename)[0] + "_512x768.mp4"
            resized_filepath = os.path.join(directory, new_filename)
            
            # Write the resized video to a file
            resized_clip.write_videofile(resized_filepath)

            # Close the clip to release resources
            video_clip.close()

# Step 2: Concatenate all the resized videos
def concatenate_resized_videos(directory, output_file):
    video_clips = []

    # Look for all files that match *_512x768.mp4
    for filename in os.listdir(directory):
        if filename.endswith("_512x768.mp4"):
            filepath = os.path.join(directory, filename)

            # Load the resized video file
            video_clip = VideoFileClip(filepath)

            # Add to the list of clips to concatenate
            video_clips.append(video_clip)

    # Concatenate all resized video clips
    if video_clips:
        final_clip = concatenate_videoclips(video_clips, method="compose")

        # Write the final concatenated clip to the output file
        final_clip.write_videofile(output_file)

        # Close the clips to release resources
        for clip in video_clips:
            clip.close()
@app.route('/resize_mp4')
def resize_mp4_route():

    # Directory containing the original *.mp4 files
    input_directory = "/home/jack/Desktop/Flask_Make_Art/static/temp_exp"

    # Output file name for the concatenated video
    output_file = "/home/jack/Desktop/Flask_Make_Art/static/temp_exp/all_asset_videos.mp4"

    # Step 1: Resize and save videos
    resize_videos(input_directory)

    # Step 2: Concatenate resized videos
    concatenate_resized_videos(input_directory, output_file)     
    return redirect(url_for('display_resources'))


archived_images_dir = 'static/novel_images'

def resize_and_crop_image(image_path):
    """Resize image to height 768 keeping aspect ratio, then center-crop to 512x768."""
    try:
        print(f"Processing image: {image_path}")
        with Image.open(image_path) as img:
            # Resize image to height 768 while maintaining the aspect ratio
            width, height = img.size
            aspect_ratio = width / height
            new_height = 768
            new_width = 512#int(aspect_ratio * new_height)

            print(f"Original size: {width}x{height}, Resizing to: {new_width}x{new_height}")

            img = img.resize((new_width, new_height), Image.LANCZOS)

            # Calculate cropping box to center-crop the image to 512x768
            #left = (new_width - 512) / 2
            #top = 0  # Since height is already 768, no need to crop vertically
            #right = (new_width + 512) / 2
            #bottom = 768

            #print(f"Cropping coordinates: left={left}, top={top}, right={right}, bottom={bottom}")

            #img = img.crop((left, top, right, bottom))

            # Save the cropped image as JPG
            new_image_path = os.path.splitext(image_path)[0] + '.jpg'
            img.convert('RGB').save(new_image_path, 'JPEG')

            print(f"Saved resized and cropped image: {new_image_path}")

            # Optionally, remove the original PNG file if it exists
            if image_path.lower().endswith('.png'):
                os.remove(image_path)
                print(f"Removed original PNG file: {image_path}")

    except Exception as e:
        print(f"Error processing {image_path}: {e}")


@app.route('/size_and_format_images')
def size_and_format_images_route():
    """Resize and format PNG/JPG images in the specified directory."""
    print("Starting image resizing and formatting...")
    for image_path in glob.glob(os.path.join(archived_images_dir, '*')):
        if image_path.lower().endswith(('.png', '.jpg')):
            print(f"Resizing image: {image_path}")
            resize_and_crop_image(image_path)
    print("Image resizing completed.")
    return redirect(url_for('img_processing_route'))


@app.route('/clean_storage', methods=['GET', 'POST'])
def clean_storage_route():
    directory_path = 'static/temp_exp'
    delete_non_X_mp4_files(directory_path)
    # Resize and format images before processing
    size_and_format_images_route()

    if request.method == 'POST':
        # Get selected images from form
        selected_images = request.form.getlist('selected_images')

        if selected_images:
            # Generate a unique video file name
            unique_id = str(uuid.uuid4())
            video_filename = os.path.join('static/image-archives', f'{unique_id}.mp4')

            # Ensure the directory for video exists
            if not os.path.exists('static/image-archives'):
                os.makedirs('static/image-archives')

            # Get dimensions from the first image to set video size
            first_image_path = os.path.join(archived_images_dir, selected_images[0])
            first_image = cv2.imread(first_image_path)
            height, width, layers = first_image.shape

            # Define the video codec and create VideoWriter object
            video = cv2.VideoWriter(video_filename, cv2.VideoWriter_fourcc(*'mp4v'), 1, (width, height))

            # Write each selected image to the video
            for image in selected_images:
                image_path = os.path.join(archived_images_dir, image)
                img = cv2.imread(image_path)
                video.write(img)

            # Release the VideoWriter object after writing the video
            video.release()

            # Remove selected images from storage
            for image in selected_images:
                image_path = os.path.join(archived_images_dir, image)
                if os.path.exists(image_path):
                    os.remove(image_path)

        return redirect(url_for('clean_storage_route'))

    # Fetch all images (JPG and PNG) in the archived directory
    images = [os.path.basename(img) for img in glob.glob(os.path.join(archived_images_dir, '*.jpg'))]
    images.extend([os.path.basename(img) for img in glob.glob(os.path.join(archived_images_dir, '*.png'))])

    # Sort images by creation time
    images = sorted(images, key=lambda x: os.path.getctime(os.path.join(archived_images_dir, x)), reverse=True)

    return render_template('img_processing.html', images=images)

@app.route('/get_videos', methods=['GET', 'POST'])
def get_videos():
    video_files = glob.glob("static/video_history/*.mp4")
    return render_template("get_videos.html", video_files=video_files)
@app.route('/upload_form')
def upload_form():
    image_directory = 'static/novel_images/'  # Directory containing the images
    images_list = []

    # Retrieve all image files from the directory
    image_files = [f for f in os.listdir(image_directory) if f.endswith('.jpg') or f.endswith('.png')]
    # List by date of creation reverse
    image_files = sorted(image_files, key=lambda x: os.path.getctime(os.path.join(image_directory, x)), reverse=True)
    return render_template('upload_form.html', video_url=None, video_type=None, image_files=image_files)


@app.route('/process_selected_images', methods=['POST'])
def process_selected_images():
    # Check if any images were uploaded
    bg_image_uploaded = request.files.get('bg_image')
    fg_image_uploaded = request.files.get('fg_image')

    # Check if any existing images were selected
    bg_image_selected = request.form.get('bg_image_selected')
    fg_image_selected = request.form.get('fg_image_selected')

    if bg_image_uploaded and bg_image_uploaded.filename != '':
        bg_filename = 'background.png'
        bg_image_uploaded.save(os.path.join('static', bg_filename))
        bg_file_path = os.path.abspath(os.path.join('static', bg_filename))
    elif bg_image_selected:
        bg_file_path = os.path.abspath(os.path.join('static/novel_images', bg_image_selected))
    else:
        return redirect(url_for('upload_form'))

    if fg_image_uploaded and fg_image_uploaded.filename != '':
        fg_filename = 'foreground.png'
        fg_image_uploaded.save(os.path.join('static', fg_filename))
        fg_file_path = os.path.abspath(os.path.join('static', fg_filename))
    elif fg_image_selected:
        fg_file_path = os.path.abspath(os.path.join('static/novel_images', fg_image_selected))
    else:
        return redirect(url_for('upload_form'))

    # Apply zoom effect and get the processed image list
    images_list = zoom_effect(bg_file_path, fg_file_path)

    if not os.path.exists('static/overlay_zooms/title'):
        os.makedirs('static/overlay_zooms/title')

    output_mp4_file = 'static/overlay_zooms/title/title_video.mp4'
    frames_per_second = 30
    create_mp4_from_images(images_list, output_mp4_file, frames_per_second)
    shutil.copy(output_mp4_file, 'static/temp_exp/title_video.mp4')
    # Create a timestamped copy of the output MP4
    file_bytime = time.strftime("%Y%m%d-%H%M%S") + ".mp4"
    shutil.copy(output_mp4_file, 'static/overlay_zooms/title/' + file_bytime)
    video_url = url_for('static', filename='overlay_zooms/title/' + file_bytime)

    return render_template('upload.html', video_url=video_url, video_type="title")

# Route to process all images in a directory (main video)@app.route('/zoom_all')
@app.route('/zoom_all')
def process_directory():
    image_directory = 'static/novel_images/'  # Directory containing the images
    images_list = []

    # Retrieve all image files from the directory
    image_files = [f for f in os.listdir(image_directory) if f.endswith('.jpg') or f.endswith('.png')]
    #list by date of creation reverse
    image_files = sorted(image_files, key=lambda x: os.path.getctime(os.path.join(image_directory, x)), reverse=True)

    # Ensure there are at least two images to work with
    if len(image_files) < 2:
        print(f"Not enough images in {image_directory} to process.")
        return render_template('upload.html', error="Not enough images to apply zoom effect.")
    
    # Iterate over image pairs
    for i in range(0, len(image_files) - 1, 2):
        # Define image paths for the current pair
        image_path1 = os.path.join(image_directory, image_files[i])
        image_path2 = os.path.join(image_directory, image_files[i + 1])

        # Apply zoom effect with two different images
        images = zoom_effect(image_path1, image_path2)
        images_list.extend(images)

        # Logging the processed images
        print(f"Processed images: {image_files[i]} and {image_files[i + 1]}")

    # Ensure the output directory exists
    if not os.path.exists('static/overlay_zooms/main'):
        os.makedirs('static/overlay_zooms/main')

    # Generate MP4 file from processed images
    output_mp4_file = 'static/overlay_zooms/main/main_video.mp4'
    frames_per_second = 30
    create_mp4_from_images(images_list, output_mp4_file, frames_per_second)

    # Save the output with a timestamp
    file_bytime = time.strftime("%Y%m%d-%H%M%S") + ".mp4"
    shutil.copy(output_mp4_file, 'static/overlay_zooms/main/' + file_bytime)
    video_url = url_for('static', filename='overlay_zooms/main/' + file_bytime)

    # Copy the video to a temporary directory
    dst = 'static/temp_exp/vopt.mp4'
    src = output_mp4_file
    shutil.copy(src, dst)

    return render_template('upload.html', video_url=video_url, video_type="main")


# Route to concatenate title and main videos
@app.route('/concatenate_videos')
def concatenate_videos_route():
    # Paths to the title and main videos
    title_video = 'static/overlay_zooms/title/title_video.mp4'
    main_video = 'static/overlay_zooms/main/main_video.mp4'

    # Check if both videos exist
    if not os.path.exists(title_video) or not os.path.exists(main_video):
        return "Both title and main videos must exist to concatenate."

    # Load video clips
    title_clip = VideoFileClip(title_video)
    main_clip = VideoFileClip(main_video)

    # Ensure both videos are resized to 512x768
    desired_resolution = (512, 768)
    title_clip = title_clip.resize(desired_resolution)
    main_clip = main_clip.resize(desired_resolution)

    # Concatenate the clips
    final_clip = concatenate_videoclips([title_clip, main_clip])

    # Prepare the output directory and file path
    output_dir = 'static/overlay_zooms/final'
    output_file = os.path.join(output_dir, 'final_video.mp4')

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Write the final video to a file
    final_clip.write_videofile(output_file, codec="libx264", fps=30)

    # Save a timestamped copy of the final video
    file_bytime = time.strftime("%Y%m%d-%H%M%S") + ".mp4"
    shutil.copy(output_file, os.path.join(output_dir, file_bytime))

    # Generate URL for the video
    video_url = url_for('static', filename=f'overlay_zooms/final/{file_bytime}')

    # Copy the final video to the temp_exp directory
    shutil.copy(output_file, 'static/temp_exp/finalX.mp4')

    # Redirect to another route after completion
    return redirect(url_for('mk_videos'))

def zoom_effect(bg_file, fg_file):
    # Open background and foreground images, convert them to RGBA for transparency
    bg = Image.open(bg_file).convert('RGBA')
    fg = Image.open(fg_file).convert('RGBA')
    
    # Get size of the background image
    SIZE = bg.size
    
    # Resize background and foreground images to the same size for consistency
    bg = bg.resize(SIZE, resample=Image.Resampling.LANCZOS)
    fg = fg.resize(SIZE, resample=Image.Resampling.LANCZOS)

    result_images = []

    # Generate zoom effect by resizing the foreground progressively
    for i in range(200):
        # Calculate progressive size scaling for foreground
        size = (int(fg.width * (i + 1) / 200), int(fg.height * (i + 1) / 200))
        
        # Resize foreground with progressive zoom
        fg_resized = fg.resize(size, resample=Image.Resampling.LANCZOS)
        
        # Apply gradual transparency (alpha) to foreground
        fg_resized.putalpha(int((i + 1) * 255 / 200))

        # Create a copy of the background image to composite the foreground onto
        result = bg.copy()

        # Calculate position to center the resized foreground
        x = int((bg.width - fg_resized.width) / 2)
        y = int((bg.height - fg_resized.height) / 2)

        # Alpha composite the resized foreground onto the background
        result.alpha_composite(fg_resized, (x, y))

        # Add the result to the list of images
        result_images.append(result)

    return result_images
# Function to create MP4 from images (unchanged)
def create_mp4_from_images(images_list, output_file, fps):
    #latest change
    #images_list = sorted(images_list, key=lambda x: os.path.getctime(os.path.join('static/novel_images', x)), reverse=True)
    image_arrays = [np.array(image) for image in images_list]
    clip = ImageSequenceClip(image_arrays, fps=fps)
    clip.write_videofile(output_file, codec="libx264", fps=fps)
    shutil.copy(output_file, 'static/temp_exp/resized_titleX.mp4')

@app.route('/archive_images')
def archive_images_route():
    source_dir = os.path.join('static', 'masks')
    dest_dir = os.path.join('static', 'archived-images/')

    # Ensure the destination directory exists
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    # Copy each image from source to destination
    for filename in os.listdir(source_dir):
        if filename.endswith('.jpg') or filename.endswith('.png'):  # Adjust extensions as needed
            src_file = os.path.join(source_dir, filename)
            dest_file = os.path.join(dest_dir, filename)
            shutil.copy(src_file, dest_file)

    return redirect(url_for('select_images'))  # Redirect to an appropriate page

@app.route('/view_masks')
def view_masks():
    masks = glob.glob('static/masks/*.jpg')
    masks = sorted(masks, key=os.path.getmtime, reverse=True)
    filenames = [os.path.basename(mask) for mask in masks]
    mask_data = zip(masks, filenames)
    return render_template('view_masks.html', mask_data=mask_data)
@app.route('/delete_mask', methods=['POST', 'GET'])
def delete_mask():
    mask_path = request.form.get('mask_path')
    if mask_path:
        try:
            os.remove(mask_path)
            print(f"Deleted mask: {mask_path}")
        except Exception as e:
            print(f"Error deleting mask: {e}")
    return redirect(url_for('view_masks'))


# Paths


@app.route('/concat_videos')
def concat_videos():
    # Define the file paths
    output_file = 'static/temp_exp/OUTPUT_PATHX.mp4'
    title_path = 'static/temp_exp/resized_title.mp4'
    main_path = 'static/temp_exp/resized_main.mp4'
    mp3_directory = 'static/mp3s'

    # Check if input files exist
    if not os.path.exists(title_path):
        print(f"Error: Title video '{title_path}' not found.")
        return "Title video not found", 400
    if not os.path.exists(main_path):
        print(f"Error: Main video '{main_path}' not found.")
        return "Main video not found", 400

    # Pick a random MP3 file from static/mp3s directory
    try:
        mp3_files = [f for f in os.listdir(mp3_directory) if f.endswith('.mp3')]
        if not mp3_files:
            print("No MP3 files found in 'static/mp3s'.")
            return "No MP3 files available", 400
        random_mp3 = os.path.join(mp3_directory, random.choice(mp3_files))
        print(f"Selected random MP3: {random_mp3}")
    except Exception as e:
        print(f"Error selecting random MP3: {e}")
        return f"Error selecting MP3: {e}", 500

    # Construct the ffmpeg command with audio
    command = [
        'ffmpeg', '-hide_banner', '-loglevel', 'info',  # Enable FFmpeg print for debugging
        '-i', title_path,
        '-i', main_path,
        '-i', random_mp3,  # Add the randomly selected MP3
        '-filter_complex', '[0:v][1:v]concat=n=2:v=1:a=0[outv]',
        '-map', '[outv]',  # Video output stream
        '-map', '2:a',     # Map the audio from the random MP3
        '-c:v', 'libx264',
        '-preset', 'medium',
        '-crf', '23',
        '-shortest',        # Use shortest flag to end when either video or audio finishes
        '-y', output_file   # Output file
    ]

    # Execute the command
    try:
        subprocess.run(command, check=True)
        print("Video concatenation and audio addition successful!")
    except subprocess.CalledProcessError as e:
        print(f"Error occurred during concatenation: {e}")
        return f"Error during concatenation: {e}", 500
    
    return redirect(url_for('mk_videos'))

def bak(filename):
    try:
        # Get the current date and time
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        #make static/backup/ if not exists
        if not os.path.exists('static/backup'):
            os.makedirs('static/backup')
        backup = glob.glob('static/backup/backup*.py')
        #if len backup > 8   delete the oldest
        if len(backup) > 8:
            backup.sort()
            os.remove(backup[0])
        # Generate a backup filename with the timestamp
        backup_filename = f"static/backup/backup_{timestamp}.py"
        
        # Copy the file to the new backup filename
        shutil.copy(filename, backup_filename)
        
        print(f"File '{filename}' successfully backed up as '{backup_filename}'.")
    except Exception as e:
        print(f"An error occurred while backing up the file: {e}")
@app.route('/moviepy')
def moviepy_route():
    return render_template('moviepy_info.html')

@app.route('/flask_info')
def flask_info_route():
    return render_template('flask_info.html')

@app.route('/moviepy_fx_info')
def moviepy_fx_route():
    return render_template('moviepy_fx.html')

@app.route('/PIL_info')
def PIL_info_route():
    return render_template('PIL_info.html')

@app.route('/crop_image')
def select_image():
    images = os.listdir(app.config['UPLOADFOLDER'])
    return render_template('viewing_page.html', images=images)

# Route for processing the image (crop, resize, sharpen)
@app.route('/processimage', methods=['POST'])
def processimage():
    try:
        image_file = request.form['image_file']
        x1 = int(request.form['x1'])
        y1 = int(request.form['y1'])
        x2 = int(request.form['x2'])
        y2 = int(request.form['y2'])

        img_path = os.path.join(app.config['UPLOADFOLDER'], image_file)
        img = cv2.imread(img_path)

        print(f"Image {image_file} loaded successfully")

        # Crop the image
        cropped_img = img[y1:y2, x1:x2]
        print(f"Image cropped: Upper-Left ({x1}, {y1}), Lower-Right ({x2}, {y2})")

        # Resize to 512x768
        resized_img = cv2.resize(cropped_img, (512, 768))
        print("Image resized to 512x768")

        # Sharpen the image using a kernel
        sharpen_kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        sharpened_img = cv2.filter2D(resized_img, -1, sharpen_kernel)
        print("Image sharpened")

        # Save the processed image temporarily for viewing
        processed_image_name = f"processed_{image_file}"
        processed_img_path = os.path.join(app.config['ARCHIVED_RESOURCES'], processed_image_name)
        cv2.imwrite(processed_img_path, sharpened_img)

        print(f"Processed image saved temporarily as {processed_image_name}")

        return render_template('viewing_page.html', processed_image=processed_image_name, images=os.listdir(app.config['UPLOADFOLDER']))
    
    except Exception as e:
        print(f"Error processing image: {e}")
        return redirect(url_for('select_image'))

# Route for saving the processed image permanently
@app.route('/saveimage', methods=['POST'])
def saveimage():
    try:
        processed_image = request.form['image_file']
        processed_img_path = os.path.join(app.config['ARCHIVED_RESOURCES'], processed_image)

        # Move the processed image to final storage if necessary
        # For now, it's already in the `archived_resources` folder

        print(f"Image {processed_image} saved successfully")

        return redirect(url_for('select_image'))
    
    except Exception as e:
        print(f"Error saving image: {e}")
        return redirect(url_for('select_image'))
    
@app.route('/fadem')
def fadem():     
        subprocess.run(['/home/jack/miniconda3/envs/cloned_base/bin/python', 'fadem'], check=True)
        return redirect(url_for('mk_videos'))
# Helper functions
def ensure_dir_exists(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
        print(f"Directory created: {dir_name}")
    else:
        print(f"Directory already exists: {dir_name}")

@app.route('/all_code')
def all_code():
    return render_template('all_code.html')

@app.route('/all_html')
def all_html():
    return render_template('all_html.html')


VIDEO_DIR = 'static/temp_exp'

@app.route('/view_videos')
def view_videos():
    # List all videos from the temp-exp folder
    videos = [f for f in os.listdir(VIDEO_DIR) if f.endswith('.mp4')]
    #videos = sorted(videos, key=os.path.getmtime, reverse=True)
    return render_template('view_video.html', videos=videos)

@app.route('/concatenate_novel', methods=['POST'])
def concatenate_novel():
    try:
        # Get the selected videos from the form
        video1_filename = request.form.get('video1')
        video2_filename = request.form.get('video2')

        if not video1_filename or not video2_filename:
            return "Please select two videos", 400

        # Full paths for the selected videos
        #video1_path = os.path.join(VIDEO_DIR, video1_filename)
        #video2_path = os.path.join(VIDEO_DIR, video2_filename)

        video1_path = video1_filename
        video2_path = video2_filename
        resize_video(video1_path, 'static/temp_exp/resized_video1.mp4')
        resize_video(video2_path, 'static/temp_exp/resized_video2.mp4')
        concatenate_videos('static/temp_exp/resized_video1.mp4', 'static/temp_exp/resized_video2.mp4', 'static/temp_exp/novel_concatenationX.mp4')
        # Redirect back to view_video.html with the result
        shutil.copy('static/temp_exp/novel_concatenationX.mp4', 'static/novel_images/novel_concatenation.mp4')
        return redirect(url_for('view_videos'))

    except Exception as e:
        return f"Error: {str(e)}", 500


def resize_video(input_file, output_file, width=512, height=768):
    """
    Resizes and pads a video to the target resolution (512x768) while maintaining aspect ratio.
    """
    try:
        subprocess.run([
            "ffmpeg", "-i", input_file, 
            "-vf", f"scale={width}:-1,pad={width}:{height}:(ow-iw)/2:(oh-ih)/2",
            "-c:a", "copy","-y", output_file
        ], check=True)
        print(f"Resized and padded {input_file} to {output_file}")
    except subprocess.CalledProcessError as e:
        print(f"Error resizing video {input_file}: {e}")

def concatenate_videos(video1, video2, output_file):
    """
    Concatenates two videos sequentially with audio using ffmpeg.
    """
    try:
        subprocess.run([
            "ffmpeg", "-i", video1, "-i", video2, 
            "-filter_complex", "[0:v][0:a][1:v][1:a]concat=n=2:v=1:a=1","-y", 
            output_file
        ], check=True)
        print(f"Concatenated videos saved as {output_file}")
    except subprocess.CalledProcessError as e:
        print(f"Error concatenating videos: {e}")

@app.route('/concatenatem', methods=['POST'])
def concatenatem():
    try:
        # Get the selected videos from the form
        video1_filename = request.form.get('video1')
        video2_filename = request.form.get('video2')

        if not video1_filename or not video2_filename:
            return "Please select two videos", 400

        # Full paths for the selected videos
        video1_path = os.path.join(VIDEO_DIR, video1_filename)
        video2_path = os.path.join(VIDEO_DIR, video2_filename)

        # Load the video files using MoviePy
        video1 = VideoFileClip(video1_path)
        video2 = VideoFileClip(video2_path)

        # Concatenate the video clips
        final_clip = concatenate_videoclips([video1, video2], method="compose")

        # Save the concatenated video back to the temp-exp directory
        output_filename = 'concatenated_video.mp4'
        output_path = os.path.join(VIDEO_DIR, output_filename)
        final_clip.write_videofile(output_path, codec="libx264", audio_codec="aac")
        #run ffmpeg command to fix video
        subprocess.run(['ffmpeg', '-i', output_path, '-c:a', 'copy','-y' ,'static/temp_exp/concatenated_videoX.mp4'], check=True)
        # Close video files to free resources
        video1.close()
        video2.close()

        # Redirect back to view_video.html with the result
        return redirect(url_for('view_videos'))

    except Exception as e:
        return f"Error: {str(e)}", 500


@app.route('/add_border')
def add_border():
    images = [f for f in os.listdir(
            'static/archived_resources/') if os.path.isfile(os.path.join('static/archived_resources/', f))]
    thumbnails = []
    for image in images:
        with Image.open('static/archived_resources/' + image) as img:
            img.thumbnail((200, 200))
            thumbnail_name = 'thumbnail_' + image
            img.save('static/thumbnails/' + thumbnail_name)
            thumbnails.append(thumbnail_name)
    return render_template('add_border.html', images=images, thumbnails=thumbnails)
    
    
@app.route('/select_border')
def select_border():
    borders = os.listdir('static/transparent_borders/')
    return render_template('select_border.html', borders=borders)
    
    
@app.route('/apply_border', methods=['POST', 'GET'])
def apply_border():
    if not os.path.exists('archived_resources'):
        os.makedirs('archived_resources')
    selected_image = request.form['image']
    selected_border = request.form['border']
    try:
        with Image.open('static/archived_resources/' + selected_image) as img:
            with Image.open('static/transparent_borders/' + selected_border) as border:
                img = img.resize(border.size)
                img.paste(border, (0, 0), border)
                final_image_name = 'final_' + selected_image
                img.save('static/final_images/' + final_image_name)
        return render_template('final_image.html', final_image=final_image_name, message='Border applied successfully.')
    except Exception as e:
        error_message = f'An error occurred: {str(e)}. Please try again.'
        return render_template('apply_border.html', image=selected_image, border=selected_border, error_message=error_message)
    
    
@app.route('/select_border_image', methods=['GET'])
def select_border_image():
    try:
        image = request.args.get('image')
        if not image:
            raise ValueError('No image selected.')
        return render_template('select_border.html', image=image, borders=os.listdir('static/transparent_borders/'))
    except Exception as e:
        error_message = f'An error occurred: {str(e)}. Please try again.'
        return render_template('add_border.html', error_message=error_message)
    
# Route for creating a text file and converting it to MP3



@app.route('/create_text_file', methods=['GET', 'POST'])
def create_text_file():
    if request.method == 'POST':
        print("Received request to create a text file.")
        # Get the text content from the textarea
        text_content = request.form.get('textarea_content')
        print(f"Text content for file: {text_content}")

        # Create the file path
        text_file_path = os.path.join('static/new_video', 'text_file.txt')

        # Write the text content to the file
        with open(text_file_path, 'w') as file:
            file.write(text_content)
        print(f"Text file created at {text_file_path}")

        return render_template('text_file_created.html', text_file_path=text_file_path)

    return render_template('create_text_file.html')

# Path to store MP3 files
MP3_DIR = 'static/audio_mp3/'
@app.route('/text_mp3', methods=['GET', 'POST'])
def text_mp3():
    if request.method == 'POST':
        print("Received POST request.")

        # Get the text from the form
        text = request.form['text']
        print(f"Text received: {text}")

        # Preserve original text and clean it up for filenames
        text0 = text
        text_clean = text.replace(" ", "")
        print(f"Cleaned text (no spaces): {text_clean}")

        # Create filenames for the MP3 and text files
        filename = MP3_DIR + text_clean[:25] + ".mp3"
        textname = "static/text/" + text_clean[:25] + ".txt"
        print(f"Generated MP3 filename: {filename}")
        print(f"Generated text file name: {textname}")

        # Save the text to a file
        with open(textname, 'w') as f:
            f.write(text0)
        print(f"Text saved to {textname}")

        # Convert text to MP3 using gTTS
        tts = gTTS(text=text0)
        tts.save(filename)
        print(f"MP3 file saved as {filename}")

        # Copy the MP3 file to a temporary location
        shutil.copy(filename, 'static/TEMP.mp3')
        print("MP3 file copied to static/TEMP.mp3")

        # Play the MP3 using pygame (optional, comment out if not needed)
        pygame.mixer.init()
        pygame.mixer.music.load(filename)
        pygame.mixer.music.play()
        print("Playing MP3 audio...")

        # Wait for the audio to finish playing
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)

        pygame.mixer.quit()
        print("Finished playing MP3, pygame exited.")

        # Return the text and list of MP3s to the template
        return render_template('text_mp3.html', text=text0, filename=filename, mp3_list=get_mp3_list())
    
    # GET request: Render form with the list of existing MP3 files
    print("Rendering form with list of existing MP3 files.")
    return render_template('text_mp3.html', mp3_list=get_mp3_list())

# Route to play an MP3 selected by the user
@app.route('/play_mp3/<filename>')
def play_mp3(filename):
    # Render the template to play the selected MP3
    return render_template('play_mp3.html', filename=filename)

# Helper function to list all MP3s in the directory
def get_mp3_list():
    mp3_list = [f for f in os.listdir(MP3_DIR) if f.endswith('.mp3')]
    #mp3_list = glob.glob(MP3_DIR+'*.mp3')
    mp3_list = sorted(mp3_list, key=os.path.getmtime, reverse=True)
    return mp3_list


@app.route('/add_audio', methods=['POST'])
def add_audio():
    # Get the video and audio file paths from the form
    video = request.form['video']
    audio = request.form['audio']

    # Define the full paths to the video and audio files
    video_path = os.path.join('static/temp_exp', video)
    audio_path = os.path.join('static/audio_mp3', audio)

    # Call function to add the audio to the video
    output_path = add_sound_to_video(video_path, audio_path)

    # Redirect to the play_video route to show the combined video
    return redirect(url_for('play_video', filename=os.path.basename(output_path)))

def add_sound_to_video(video_path, audio_path):
    # Load the video clip
    video_clip = VideoFileClip(video_path)

    # Load the audio clip
    audio_clip = AudioFileClip(audio_path)

    # Set the audio for the video
    video_with_audio = video_clip.set_audio(audio_clip)

    # Define the output path for the new video
    output_path = video_path.replace('.mp4', '_audio.mp4')

    # Write the video with audio to a new file
    video_with_audio.write_videofile(output_path, codec='libx264', audio_codec='aac')

    return output_path

#redundent 
@app.route('/combinez', methods=['POST'])
def combinez_video_audio():
    image_file = request.form.get('image_path')
    audio_file = request.form.get('audio_path')

    if not image_file or not audio_file:
        logit("No image or audio file selected")
        return jsonify({'error': 'No image or audio selected'}), 400

    logit(f"Selected image: {image_file}")
    logit(f"Selected audio: {audio_file}")

    # Construct full paths for the image and audio
    image_path = os.path.join('static/archived_resources', image_file)
    audio_path = os.path.join('static/audio_mp3', audio_file)

    # Generate output filename
    output_filename = f"output_{image_file.rsplit('.', 1)[0]}_{audio_file.rsplit('.', 1)[0]}.mp4"
    output_path = os.path.join('static/temp_exp', output_filename)

    try:
        logit(f"Processing image: {image_path} and audio: {audio_path}")

        # Create video from image and audio
        output_path = add_sound_to_image(image_path, audio_path)
        if output_path:
            return jsonify({'success': True, 'output_video': output_path})
        else:
            return jsonify({'error': ' 3961'}), 500

    except Exception as e:
        logit(f"Error processing video and audio: {str(e)}")
        return jsonify({'error': str(e)}), 500



@app.route('/add_sound_image', methods=['GET'])
def render_add_sound_form():
    # Get list of audio and image files
    audio_files = [f for f in os.listdir('static/audio_mp3') if f.endswith('.mp3')]
    image_files = [f for f in os.listdir('static/archived_resources') if f.endswith(('.png', '.jpg'))]
    random.shuffle(image_files)
    image_files = image_files[:50]  # Limit to 50 images
    return render_template('sound_to_image.html', audio_files=audio_files, image_files=image_files)

@app.route('/add_sound_square', methods=['GET'])
def render_add_sound_square_form():
    # Get list of audio and image files
    audio_files = [f for f in os.listdir('static/output') if f.endswith('.mp3')]
    # sort audio by date reversed
    audio_files.sort(key=lambda x: os.path.getmtime(os.path.join('static/output', x)), reverse=True)
    image_files = [f for f in os.listdir('static/square') if f.endswith(('.png', '.jpg'))]
    random.shuffle(image_files)
    image_files = image_files[:50]  # Limit to 50 images
    return render_template('sound_to_square.html', audio_files=audio_files, image_files=image_files)


from moviepy.editor import ImageClip, AudioFileClip

@app.route('/play_video/<filename>')
def play_video(filename):
    # This route will display the newly generated video
    return render_template('play_video.html', video_file=filename)





# Global variable to store the process ID of the running FFmpeg process
ffmpeg_pid = None

# Define the path for your video output directory
video_output_dir = "/home/jack/Desktop/Flask_Make_Art/static/videos/"

@app.route('/start_ffmpeg', methods=['POST', 'GET'])
def start_ffmpeg():
    global ffmpeg_pid
    try:
        # Define the FFmpeg command
        ffmpeg_cmd = [
            'ffmpeg',
            '-hide_banner',
            '-f', 'pulse',
            '-i', 'alsa_output.pci-0000_00_1b.0.analog-stereo.monitor',
            '-f', 'x11grab',
            '-framerate', '30',
            '-video_size', '1366x760',
            '-i', ':0.0',
            '-c:v', 'libx264',
            '-preset', 'ultrafast',
            '-pix_fmt', 'yuv420p',
            '-c:a', 'aac',
            '-strict', '-2',
            '-g', '120',
            '-f', 'flv',
            '-y', f"{video_output_dir}$(date +'%Y-%m-%d_%H-%M-%S')_video3.flv"
        ]
        
        # Start the FFmpeg process
        process = subprocess.Popen(ffmpeg_cmd)
        ffmpeg_pid = process.pid
        
        #print.debug(f"Started FFmpeg process with PID: {ffmpeg_pid}")
        return jsonify({'message': 'FFmpeg started', 'pid': ffmpeg_pid}), 200
    except Exception as e:
        #print.error(f"Error starting FFmpeg: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/stop_ffmpeg', methods=['POST', 'GET'])
def stop_ffmpeg():
    global ffmpeg_pid
    if ffmpeg_pid:
        try:
            # Stop the FFmpeg process using the stored PID
            os.kill(ffmpeg_pid, signal.SIGTERM)
            #print.info(f"Stopped FFmpeg process with PID: {ffmpeg_pid}")
            ffmpeg_pid = None  # Reset PID
            return jsonify({'message': 'FFmpeg stopped'}), 200
        except Exception as e:
            #print.error(f"Error stopping FFmpeg: {str(e)}")
            return jsonify({'error': str(e)}), 500
    else:
        #print.warning("No FFmpeg process is running")
        return jsonify({'message': 'No FFmpeg process to stop'}), 400

# Directories
CONTENT_DIR = 'static/archived_resources'
STYLE_DIR = 'static/style_resources'
RESULT_DIR = 'static/novel_resources'

if not os.path.exists(CONTENT_DIR):
    os.makedirs(CONTENT_DIR)
if not os.path.exists(RESULT_DIR):
    os.makedirs(RESULT_DIR)
if not os.path.exists(STYLE_DIR):
    os.makedirs(STYLE_DIR)

# Load the TensorFlow Hub model
hub_model = hub.load("http://0.0.0.0:8000/magenta_arbitrary-image-stylization-v1-256_2.tar.gz")

# Helper functions
def tensor_to_image(tensor):
    tensor = tensor * 255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor) > 3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    return Image.fromarray(tensor)

def load_img(path_to_img):
    max_dim = 512
    img = tf.io.read_file(path_to_img)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    
    shape = tf.cast(tf.shape(img)[:-1], tf.float32)
    long_dim = max(shape)
    scale = max_dim / long_dim

    new_shape = tf.cast(shape * scale, tf.int32)
    img = tf.image.resize(img, new_shape)
    img = img[tf.newaxis, :]
    return img

@app.route('/styling', methods=['GET', 'POST'])
def styling():
    if request.method == 'POST':
        content_image_name = request.form['content_image']
        style_image_name = request.form['style_image']

        # Prepare paths
        content_path = os.path.join(CONTENT_DIR, content_image_name)
        style_path = os.path.join(STYLE_DIR, style_image_name)

        # Load content and style images
        content_image = load_img(content_path)
        style_image = load_img(style_path)

        # Stylize the image
        stylized_image = hub_model(tf.constant(content_image), tf.constant(style_image))[0]
        result_image = tensor_to_image(stylized_image)

        # Save the result image
        timestr = time.strftime("%Y%m%d-%H%M%S")
        result_filename = f"Result_{timestr}.jpg"
        result_path = os.path.join(RESULT_DIR, result_filename)
        tempfile = f"static/archived_resources/tmp_result.jpg"
        result_image.save(tempfile)
        convert_to_mellow_colors(tempfile)
        shutil.copy(tempfile, result_path)
        #copy to a uuid file in static/novel_resources
        uid = str(uuid.uuid4())
        shutil.copy(tempfile, f"static/novel_resources/{uid}.jpg")
        return render_template('styling.html', content_images=os.listdir(CONTENT_DIR)[0:20],style_images=os.listdir(STYLE_DIR)[0:20], 
                               result_image=tempfile)

    # List content and style images
    content_images = os.listdir(CONTENT_DIR)
    style_images = os.listdir(STYLE_DIR)
    #shuffle the images
    random.shuffle(content_images)
    random.shuffle(style_images)
    #display only 20 images
    content_images = content_images[:20]
    style_images = style_images[:20]
    return render_template('styling.html', content_images=content_images, style_images=style_images)
# Define the 9 colors
palette = {
    "red": [255, 0, 0],
    "green": [0, 255, 0],
    "blue": [0, 0, 255],
    "cyan": [0, 255, 255],
    "magenta": [255, 0, 255],
    "yellow": [255, 255, 0],
    "black": [0, 0, 0],
    "white": [255, 255, 255],
    "gray": [128, 128, 128]
}

# Convert a pixel to the closest color in the palette
def closest_color(pixel):
    distances = {}
    for color_name, color_value in palette.items():
        distance = np.linalg.norm(np.array(pixel) - np.array(color_value))
        distances[color_name] = distance
    closest = min(distances, key=distances.get)
    return palette[closest]

def convert_to_mellow_colors(image_path):
    # Open the image
    image = Image.open(image_path)
    
    # Step 1: Enhance the saturation moderately (factor 1.5)
    enhancer = ImageEnhance.Color(image)
    enhanced_image = enhancer.enhance(1.5)
    
    # Step 2: Convert to RGB if not already
    enhanced_image = enhanced_image.convert("RGB")
    
    # Step 3: Convert to numpy array for manipulation
    img_array = np.array(enhanced_image)
    
    # Step 4: Apply color mapping (map each pixel to the closest color in our 9-color palette)
    mellow_image = np.apply_along_axis(closest_color, 2, img_array)
    
    # Step 5: Convert back to an image
    result_image = Image.fromarray(np.uint8(mellow_image))
    
    # Step 6: Save the result image with uuid
    uid = str(uuid.uuid4())
    output_uid = f'static/archived_resources/{uid}_output_primary_colors.jpg'
    outputuid = f'static/novel_images/{uid}_output_primary_colors.jpg'
    output_path = 'static/archived_resources/enhanced.jpg'
    result_image.save(output_path)
    shutil.copy(output_path, output_uid)
    shutil.copy(output_path, outputuid)
    print(f'Saved output to: {output_path}')
    return output_uid

# Define the static directories
STATIC_FOLDER = 'static'
ARCHIVE_FOLDER = os.path.join(STATIC_FOLDER, 'archived_resources')

# Ensure the archive folder exists
if not os.path.exists(ARCHIVE_FOLDER):
    os.makedirs(ARCHIVE_FOLDER)

@app.route('/copy')
def copy():
    # List directories under static folder (except archived_resources)
    directories = [
    d for d in os.listdir(STATIC_FOLDER) 
    if os.path.isdir(os.path.join(STATIC_FOLDER, d)) 
    and d != 'archived_resources' 
    and d.strip().endswith('resources')  # Ensure trailing spaces are removed
]
    #sort the directories
    print(f"Directories: {directories}")
    # sorted by creation reversed order
    directories.sort(key=lambda x: os.path.getctime(os.path.join(STATIC_FOLDER, x)), reverse=True)
    static_gallery_data = load_static_gallery_data()
    if not static_gallery_data:
        print('No static gallery data found. Creating new static gallery data.')
        scan_data = load_scan_data()
        gallery_data = select_random_images(scan_data)
        static_gallery_data = copy_images_to_static(gallery_data)
        save_static_gallery_data(static_gallery_data)
        videos_data, corrupt_videos = scan_videos()
    return render_template('copy.html', directories=directories, resources=resource_images(),data=static_gallery_data,)
    #return render_template('copy.html', directories=directories, resources=resource_images(),data=static_gallery_data,videos=videos_data, corrupt_videos=corrupt_videos)

@app.route('/view_images', methods=['POST'])
def view_images():
    # Get selected directory from the form
    selected_dir = request.form.get('directory')
    directory_path = os.path.join(STATIC_FOLDER, selected_dir)

    # List image files in the selected directory (only common image formats)
    images = [f for f in os.listdir(directory_path)
              if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif'))]
    # shuffle and display only 20 images
    random.shuffle(images)
    images = images[:20]
    return render_template('display_directories.html', images=images, selected_dir=selected_dir)

@app.route('/copy_image', methods=['POST'])
def copy_image():
    # Get selected images and directory from the form
    selected_images = request.form.getlist('selected_images')
    selected_dir = request.form.get('selected_dir')

    source_dir = os.path.join(STATIC_FOLDER, selected_dir)

    # Copy each selected image to the archive folder
    for image in selected_images:
        source_path = os.path.join(source_dir, image)
        destination_path = 'static/archived_resources/' + image
        print(f"Copying {image} to {destination_path}")
        shutil.copy2(source_path, destination_path)
    
    return redirect(url_for('copy'))

def is_server_running(port):
    """Check if there is a process listening on the specified port."""
    for proc in psutil.process_iter(['pid', 'name', 'connections']):
        try:
            # Check if 'connections' is available
            if proc.info['connections'] is not None:
                for conn in proc.info['connections']:
                    if conn.laddr.port == port:
                        return True
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            # Handle cases where the process has terminated or access is denied
            continue
    return False

def start_tensorflow_server():
    server_port = 8000  # Adjust this to the port your server uses
    if is_server_running(server_port):
        print(f"TensorFlow server is already running on port %d , {server_port}")
        return  # Exit if the server is already running

    try:
        # Path to your server script
        server_script = "python -m http.server 8000 --directory /mnt/HDD500/TENSORFLOW/models/"
        
        # Start the server as a subprocess
        print("Starting TensorFlow server...")
        process = subprocess.Popen(['bash', server_script], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Optional: log output
        print("TensorFlow server started with PID: %s", process.pid)

    except Exception as e:
        print(f"Failed to start TensorFlow server: {e}")

# Paths to video and audio directories
VIDEO_DIR = 'static/temp_exp'
AUDIO_DIR = 'static/audio_mp3'
OUTPUT_DIR = 'static/output_videos'
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR) # Create the output directory if it doesn't exist 


@app.route('/sound_2_video')
def sound_2_video():
    print("Serving main page with video and audio files")
    
    # Get video and audio file lists
    video_files = os.listdir(VIDEO_DIR)
    audio_files = os.listdir(AUDIO_DIR)
    
    # Filter only supported video/audio files (e.g., .mp4 and .mp3)
    video_files = [f for f in video_files if f.endswith('.mp4')]
    audio_files = [f for f in audio_files if f.endswith('.mp3')]
    
    print(f"Found video files: {video_files}")
    print(f"Found audio files: {audio_files}")
    
    return render_template('sound_to_video.html', video_files=video_files, audio_files=audio_files)




@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory('static', filename)

# Path to the uploads and output folders
UPLOAD_FOLDER = 'static/uploads/'
OUTPUT_FOLDER = 'static/uploads/'

@app.route('/video_edit')
def video_edit():
    return render_template('video_edit.html')

@app.route('/trim-video', methods=['POST'])
def trim_video():
    start_time = float(request.form['startTime'])
    end_time = float(request.form['endTime'])
    input_video = 'static/video_resources/use.mp4'
    output_video = os.path.join(OUTPUT_FOLDER, 'trimmed_video.mp4')

    try:
        # Log the trimming process
        print(f"Trimming video from {start_time} to {end_time}")

        # Use MoviePy to trim the video
        ffmpeg_extract_subclip(input_video, start_time, end_time, targetname=output_video)
        return redirect(url_for('video_edit'))

    except Exception as e:
        print(f"Error trimming video: {str(e)}")
        return f"Error: {str(e)}", 500
@app.route('/reverse_video', methods=['POST','GET'])
def reverse_video():
    inc = str(random.randint(0,99))
    input_video = '/home/jack/Desktop/Flask_Make_Art/static/video_resources/forward.mp4'
    video_output_dir = 'static/temp_exp'
    # Define the FFmpeg command
    _cmd = [
        'ffmpeg',
        '-hide_banner',
        '-i', f'{input_video}',
        '-vf', 'reverse',
        '-af', 'areverse', 
        '-y', f'static/temp_exp/reverse.mp4'
        ]
        
    # Start the FFmpeg process
    subprocess.run(_cmd, check=True)
    return redirect(url_for('video_edit'))

# Define path for videos and temp export
VIDEO_DIR = os.path.join('static', 'temp_exp')
TEMP_EXP_DIR = os.path.join('static', 'videos')

# Function to list all video files in a directory recursively
def list_videos_recursive(directory):
    video_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(('X.mp4', 'X.avi', 'X.mkv')):  # Add more formats if needed
                video_files.append(os.path.join(root, file))
    return video_files

@app.route('/ffmpeg')
def ffmpeg():
    # List all videos from the static/videos/ directory
    videos = list_videos_recursive(VIDEO_DIR)
    processed_videos = os.listdir(TEMP_EXP_DIR)  # List processed videos
    return render_template('ffmpeg.html', videos=videos, processed_videos=processed_videos)

@app.route('/process_ffmpeg', methods=['POST'])
def process_ffmpeg():
    if request.method == 'POST':
        video_path = request.form['video_path']
        start_time = request.form['start_time']
        duration = request.form['duration']
        output_filename = request.form['output_filename']

        # Make sure the output filename ends with .mp4
        if not output_filename.endswith('.mp4'):
            output_filename += '.mp4'

        # Set the full path for the output file
        output_path = os.path.join(TEMP_EXP_DIR, output_filename)

        # Build the ffmpeg command
        command = [
            'ffmpeg',
            '-i', video_path,  # input video path
            '-ss', start_time,  # start time
            '-t', duration,     # duration
            '-c:a', 'copy',
            '-y',       # copy codec to avoid re-encoding
            output_path         # output video path
        ]

        # Run the ffmpeg command
        try:
            print(f"Running command: {' '.join(command)}")
            subprocess.run(command, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error during processing: {e}")
            return f"Error during processing: {e}"

        # Redirect to the ffmpeg page to refresh the processed videos
        return redirect(url_for('ffmpeg'))

@app.route('/temp_exp/<filename>')
def send_processed_video(filename):
    return send_from_directory(TEMP_EXP_DIR, filename)

@app.route('/edit_description', methods=['GET', 'POST'])
def edit_description():
    # Open the 'the_description.html' file and read its contents
    with open('templates/application_overview.html', 'r') as file:
        description = file.read()

    return render_template('edit_about.html', description=description)

    
@app.route('/save_description', methods=['POST'])
def save_description():
    # Retrieve the updated description from the form submission
    updated_description = request.form['description']

    # Open the 'the_description.html' file in write mode and save the updated description
    with open('templates/application_overview.html', 'w') as file:
        file.write(updated_description)
        return redirect(url_for('edit_description'))



def degrade_color(color):
    """Decreases color intensity to simulate ink fade."""
    return tuple(max(0, c - 30) for c in color)

def draw_blob(img, center, size, color):
    """Draws a blob (circle) at the specified point."""
    for _ in range(size):
        # Random offset within a small range to create irregularity
        offset_x = random.randint(-size // 2, size // 2)
        offset_y = random.randint(-size // 2, size // 2)
        blob_point = (center[0] + offset_x, center[1] + offset_y)
        if 0 <= blob_point[0] < img.size[0] and 0 <= blob_point[1] < img.size[1]:
            img.putpixel(blob_point, color)

def processr_image(seed_count, seed_max_size, imgsize=(510, 766), count=0):
    """Generates a Rorschach-style inkblot."""
    margin_h, margin_v = 60, 60
    color = (0, 0, 0)
    img = Image.new("RGB", imgsize, "white")
    
    for seed in range(seed_count):
        point = (random.randrange(0 + margin_h, imgsize[0] // 2),
                 random.randrange(0 + margin_v, imgsize[1] - margin_v))

        # Random blob size
        blob_size = random.randint(10, seed_max_size)
        draw_blob(img, point, blob_size, color)

    # Symmetry: Flip left half onto right half
    cropped = img.crop((0, 0, imgsize[0] // 2, imgsize[1]))
    flipped = cropped.transpose(Image.FLIP_LEFT_RIGHT)
    img.paste(flipped, (imgsize[0] // 2, 0))

    # Apply a blur to smooth the inkblot edges
    blurred_img = img.filter(ImageFilter.GaussianBlur(radius=15))

    # Convert to grayscale and find the mean for thresholding
    im_grey = blurred_img.convert('L')
    mean = np.mean(np.array(im_grey))

    # Binarize the blurred image based on the mean threshold
    image_array = np.array(im_grey)
    binary_image = np.where(image_array > mean, 255, 0).astype(np.uint8)

    # Save the binary image
    temp_filename = "static/images/tmmmp.png"
    cv2.imwrite(temp_filename, binary_image)

    # Expand the border and save the final image
    final_filename = time.strftime("static/archived-images/GOODblots%Y%m%d%H%M%S.png")
    ImageOps.expand(Image.fromarray(binary_image), border=1, fill='white').save(final_filename)

    # Create and save the inverted image (black and white swapped)
    inverted_image = np.where(binary_image == 255, 0, 255).astype(np.uint8)
    inverted_filename = time.strftime("static/archived-images/INVERTEDblots%Y%m%d%H%M%S.png")
    ImageOps.expand(Image.fromarray(inverted_image), border=1, fill='black').save(inverted_filename)
    #copy blots to static/blot_resources/
    shutil.copy(final_filename, "static/blot_resources/")
    shutil.copy(inverted_filename, "static/blot_resources/")
    return final_filename, inverted_filename

@app.route('/inkblot')
def rorschach():
    ensure_dir_exists("static/images")
    ensure_dir_exists("static/blot_resources")

    # Generate the inkblots
    inkblot_images = []
    for count in range(2):  # Generate 2 inkblots as an example
        seed_count = random.randint(6, 10)
        seed_max_size = random.randint(100, 400)  # Smaller sizes to generate blobs
        final_image, inverted_image = processr_image(seed_count, seed_max_size, count=count)
        inkblot_images.append({'normal': final_image, 'inverted': inverted_image})

    # Pass the image paths to the template
    return render_template('Rorschach.html', inkblot_images=inkblot_images)
def process_image(image_path):
    img = imread(image_path)
    labels = segmentation.slic(img, compactness=30, n_segments=400)
    g = future.graph.rag_mean_color(img, labels)

    def weight_boundary(graph, src, dst, n):
        default = {'weight': 0.0, 'count': 0}
        count_src = graph[src].get(n, default)['count']
        count_dst = graph[dst].get(n, default)['count']
        weight_src = graph[src].get(n, default)['weight']
        weight_dst = graph[dst].get(n, default)['weight']
        count = count_src + count_dst
        return {
            'count': count,
            'weight': (count_src * weight_src + count_dst * weight_dst) / count
        }

    def merge_boundary(graph, src, dst):
        pass

    labels2 = future.graph.merge_hierarchical(labels, g, thresh=0.08, rag_copy=False,
                                              in_place_merge=True,
                                              merge_func=merge_boundary,
                                              weight_func=weight_boundary)

    out = color.label2rgb(labels2, img, kind='avg')
    
    # Save the processed image
    output_filename = os.path.join(app.config['UPLOAD_FOLDER'], f"processed_{time.time()}.png")
    imsave(output_filename, out)
    return output_filename

@app.route('/upload_mp4_video')
def upload_mp4_video():
    return render_template('upload_mp4.html')
@app.route('/upload_mp4', methods=['POST'])
def upload_mp4():
    uploaded_file = request.files['videoFile']
    if uploaded_file.filename != '':
        # Save the uploaded file to a directory or process it as needed
        # For example, you can save it to a specific directory:
        uploaded_file.save('static/video_resources/forward.mp4')
        #                   /' + uploaded_file.filename)
        VIDEO='static/video_resources/forward.mp4'
        return render_template('upload_mp4.html',VIDEO=VIDEO)
    else:
        VIDEO='static/video_resources/forward.mp4'
        return render_template('upload_mp4.html',VIDEO=VIDEO)

@app.route('/view_files')
def view_files():
    masks = glob.glob('static/temp_exp/*.mp4')
    masks = sorted(masks, key=os.path.getmtime, reverse=True)
    filenames = [os.path.basename(mask) for mask in masks]
    mask_data = zip(masks, filenames)
    return render_template('view_files.html', videos=masks)
print("TEST LOG")
@app.route('/delete_file', methods=['POST', 'GET'])
def delete_file():
    mask_path = request.form.get('mask_path')
    print(f'XXXXXXXX {mask_path}')
    if mask_path:
        try:
            os.remove(mask_path)
            print(f"Deleted mask: {mask_path}")
        except Exception as e:
            print(f"Error deleting mask: {e}")
    return redirect(url_for('view_files'))





@app.route('/notes')
def notes():
    with open('static_new/text/notes_app.txt') as f:
        text = f.read()
        #paragraph = text.split('----------')
        #search the paragraph for "uploads"

    return render_template('note_app_note.html', text=text)
 # split at the line "----------" and return the second part

@app.route('/notes_index', methods=['POST', 'GET'])
def notes_index():
    if request.method == 'POST':
        search_term = request.form.get('search', '').strip()
        if search_term:
            with open('static_new/text/notes_app.txt', 'r') as f:
                text = f.read()
                paragraphs = text.split('----------')

                # Filter paragraphs that contain the search term
                matching_paragraphs = [p for p in paragraphs if search_term in p]

            if matching_paragraphs:
                print(f"Matching Paragraphs:  {matching_paragraphs}")
                return render_template('notes_app.html', text=matching_paragraphs)
            else:
                return render_template('notes_app.html', text=["No matching results."])
        else:
            return render_template('notes_app.html', text=["Enter a search term."])

    return render_template('notes_app.html', text=[])

@app.route('/search_notes', methods=['POST', 'GET'])
def search_notes():
    if request.method == 'POST':
        search_term = request.form.get('search', '').strip()
        if search_term:
            with open('static_new/text/notes_app.txt', 'r') as f:
                text = f.read()
                paragraphs = text.split('----------')

                # Filter paragraphs that contain the search term
                matching_paragraphs = [p for p in paragraphs if search_term in p]

            if matching_paragraphs:
                print(f"Matching Paragraphs: , {matching_paragraphs}")
                return render_template('notes_app.html', text=matching_paragraphs)
            else:
                return render_template('notes_app.html', text=["No matching results."])
        else:
            return render_template('notes_app.html', text=["Enter a search term."])

    return render_template('notes_app.html', text=[])

# Function to add ten dashes before and after the content
def format_content(content):
    separator = '----------\n'  # Define the separator
    formatted_content = f'{separator}{content.strip()}'  # Add separator before the content
    return formatted_content

@app.route('/append_notes', methods=['POST', 'GET'])
def append_notes():
    if request.method == 'POST':
        new_content = request.form.get('new_content', '').strip()
        if new_content:
            formatted_content = format_content(new_content)  # Format the content
            with open('static_new/text/notes_app.txt', 'a') as f:
                f.write(formatted_content)
            render_template('notes_app.html')
        else:
            return 'No content to append'

    return render_template('append_notes_app.html')
@app.route('/edit_notes', methods=['GET', 'POST'])
def edit_notes():
    if request.method == 'POST':
        content = request.form.get('content', '').strip()
        write_notes(content)
        return redirect(url_for('notes'))
    else:
        text = read_notes()
        return render_template('edit_notes.html', text=text)

@app.route('/note_index', methods=['POST', 'GET'])
def note_index():
    if request.method == 'POST':
        search_term = request.form.get('search', '').strip()
        if search_term:
            text = read_notes()
            paragraphs = text.split('----------')

            # Filter paragraphs that contain the search term
            matching_paragraphs = [p for p in paragraphs if search_term in p]

            if matching_paragraphs:
                print(f"Matching Paragraphs: , {matching_paragraphs}")
                return render_template('notes_app.html', text=matching_paragraphs)
            else:
                return render_template('notes_app.html', text=["No matching results."])
        else:
            return render_template('notes_app.html', text=["Enter a search term."])

    return render_template('notes_app.html', text=[])
# Function to read the contents of the notes file
def read_notes():
    with open('static_new/text/notes_app.txt', 'r') as f:
        return f.read()
        

# Image processing function


@app.route('/open_doors', methods=['GET', 'POST'])
def open_doors_route():
    subprocess.run(['/home/jack/miniconda3/envs/cloned_base/bin/python', 'open_door_directory.py'], check=True)
    return redirect(url_for('mk_videos'))


@app.route('/three_doors', methods=['GET', 'POST'])
def three_doors():
    if request.method == 'POST':
        # Retrieve selected images from the form
        top_path = request.form.get('top_image')
        foreground_path = request.form.get('center_image')
        background_path = request.form.get('bottom_image')

        if not top_path or not foreground_path or not background_path:
            return "Please select one top image, one center image, and one bottom image."

        # Redirect to the 'slide_apart_animation' route with selected images
        return redirect(url_for('slide_apart_animation', top_path=top_path, foreground_path=foreground_path, background_path=background_path))
    image_paths = get_image_paths()
    return render_template('three_doors.html', image_paths=image_paths)


def resize_if_needed(clip, target_size=(512, 768)):
    """Resizes the clip if it is not already the target size."""
    if clip.size != target_size:
        print(f"Image size {clip.size} is not {target_size}, resizing...")
        return clip.resize(target_size)
    print(f"Image is already the correct size: {clip.size}")
    return clip

@app.route('/slide_apart_animation', methods=['GET'])
def slide_apart_animation():
    # Retrieve the image paths passed as arguments in the URL
    top_path = request.args.get('top_path')
    foreground_path = request.args.get('foreground_path')
    background_path = request.args.get('background_path')
    
    # Check if the paths were passed correctly
    if not top_path or not foreground_path or not background_path:
        return "Top, foreground, and background images are required."

    print("Loading images...")

    # Load images using moviepy
    top_image = mp.ImageClip(top_path).set_duration(1)  # Show the top image for 1 second
    foreground = mp.ImageClip(foreground_path)
    background = mp.ImageClip(background_path)

    # Resize foreground and background if needed
    print("Verifying and resizing images if necessary...")
    foreground = resize_if_needed(foreground)
    background = resize_if_needed(background)

    # Split the foreground into two halves
    print("Splitting the foreground image into two halves.")
    left_half = foreground.crop(x1=0, y1=0, x2=256, y2=768)
    right_half = foreground.crop(x1=256, y1=0, x2=512, y2=768)

    # Define the sliding effect
    def make_frame(t):
        slide_distance = min(256, int(256 * t / 3))  # Adjust slide distance according to time

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
    animation = mp.VideoClip(make_frame, duration=3)

    # Now concatenate the title image and the sliding animation
    print("Concatenating the title image and the animation...")

    # Concatenate the top image and the sliding animation
    final_clip = mp.concatenate_videoclips([top_image, animation])

    # Optionally write the output video file
    final_clip.write_videofile("static/temp_exp/slide_apart_with_titleX.mp4", fps=24)

    return redirect(url_for('mk_videos'))

@app.route('/square_transition', methods=['GET', 'POST'])
def square_transition_route():
    subprocess.run(['/home/jack/miniconda3/envs/cloned_base/bin/python', 'square_transition'], check=True)
    return redirect(url_for('mk_videos'))
@app.route('/circular_transition', methods=['GET', 'POST'])
def circular_transition_route():
    subprocess.run(['/home/jack/miniconda3/envs/cloned_base/bin/python', 'circular_transition'], check=True)
    return redirect(url_for('mk_videos'))

# Create frames and keepers directories if they don't exist
if not os.path.exists('static/frames'):
    os.mkdir('static/frames')
if not os.path.exists('static/keepers_resourses'):
    os.mkdir('static/keepers_resourses')

# Function to extract frames from MP4 using OpenCV
def extract_frames(video_path, output_folder):
    cap = cv2.VideoCapture(video_path)
    count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imwrite(os.path.join(output_folder, f'frame_{count}.jpg'), frame)
        count += 1
    cap.release()
# ----------------------- VIDEO ROUTES -----------------------
def limit_backups(source_dir='static/novel_images',max_files = 15):
    backup_dir = 'static/backups_resources'
    
    
    # Ensure backup directory exists
    os.makedirs(backup_dir, exist_ok=True)
    
    # Get all files in the source directory, sorted by modification time (oldest first)
    files = sorted(
        [f for f in os.listdir(source_dir) if os.path.isfile(os.path.join(source_dir, f))],
        key=lambda f: os.path.getmtime(os.path.join(source_dir, f))
    )
    
    # If there are more than the max allowed, move the oldest files to the backup directory
    if len(files) > max_files:
        files_to_backup = files[:-max_files]  # Select all but the last 15 files
        
        for file in files_to_backup:
            file_path = os.path.join(source_dir, file)
            # Create a unique backup filename with a timestamp, preserving the extension
            file_extension = os.path.splitext(file)[1]
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_filename = f"{os.path.splitext(file)[0]}_{timestamp}{file_extension}"
            backup_file_path = os.path.join(backup_dir, backup_filename)
            
            # Move the file to the backup directory
            shutil.move(file_path, backup_file_path)
            print(f"Moved {file} to {backup_file_path}")

#delete images in 'static/keepers_resourses/
def keepers_resourses():
    for file in os.listdir('static/keepers_resourses'):
        file_path = os.path.join('static/keepers_resourses', file)
        if os.path.isfile(file_path):
            os.remove(file_path)

# Function to delete all files in 'static/frames/'
def delete_frames():
    frames_dir = 'static/frames'
    for file in os.listdir(frames_dir):
        file_path = os.path.join(frames_dir, file)
        if os.path.isfile(file_path):
            os.remove(file_path)
           
# Route to display video selection and form submission
@app.route('/get_frames', methods=['GET', 'POST'])
def get_frames():
    delete_frames()
    #video_dir = '/home/jack/Desktop/HDD500/Image_Retriever/static/videos/'
    video_dir = '/home/jack/Desktop/Flask_Make_Art/static/videos'
    keepers_resourses()
    # Use glob to find all .mp4 files in the directory
    video_files = glob.glob(os.path.join(video_dir, '*.mp4'))
    # sort the files by modification time
    video_files = sorted(video_files, key=os.path.getmtime, reverse=True)    
    # Get only the filenames (basename)
    video_files = [os.path.basename(video) for video in video_files]  
    video_images= glob.glob('static/keepers_resourses/*.jpg')  
    video_images = sorted(video_images, key=os.path.getmtime, reverse=True)                                                          # post the images in reverse order      
    return render_template('copy_frames.html', video_files=video_files, video_images=video_images)

# Route to handle form submission and extract frames from the selected video
@app.route('/process_frames', methods=['POST'])
def process_video():
    # Get selected video from form in copy_frames.html
    video_filename = request.form.get('video')
    # This is the location of the archive videos 
    #video_dir = '/home/jack/Desktop/HDD500/Image_Retriever/static/videos/'
    video_dir = '/home/jack/Desktop/Flask_Make_Art/static/videos'
    video_path = os.path.join(video_dir, video_filename)
    output_folder = 'static/frames'

    # Clear frames folder before extracting new frames
    for filename in os.listdir(output_folder):
        file_path = os.path.join(output_folder, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)

    # Extract frames from the selected video
    extract_frames(video_path, output_folder)

    # List all frames in the frames directory
    frames = os.listdir(output_folder)
    return render_template('copy_frames.html', frames=frames)

# Route to handle form submission for image deletion
@app.route('/add_frames', methods=['POST', 'GET'])
def add_frames_route():
    selected_images = request.form.getlist('image')
    for image in selected_images:
        id_ = str(uuid.uuid4())
        source = os.path.join('static/frames', image)
        destination1 = os.path.join('static/keepers_resourses', id_ + image)
        shutil.copy(source, destination1)
        destination2 = os.path.join('static/archived-images', id_ + image)
        shutil.copy(source, destination2)
        destination3 = os.path.join('static/novel_images', id_ + image)
        shutil.copy(source, destination3)
        destination4 = os.path.join('static/novel_images', id_ + image)
        shutil.copy(source, destination4)
    limit_backups(source_dir='static/novel_images',max_files=40)
    limit_backups(source_dir = 'static/novel_images')            
    return redirect(url_for('get_frames'))


# ----------------------- END VIDEO ROUTES -----------------------    
# ----------------------- SEARCH ROUTES -----------------------
# Define the directory where your templates are stored
TEMPLATE_FOLDER = 'templates'

@app.route('/search_templates', methods=['GET', 'POST'])
def search_templates_route():
    results = []
    search_term = ''

    if request.method == 'POST':
        search_term = request.form.get('search_term', '').strip().lower()
        if search_term:
            # Search through all the HTML files in the template folder
            for root, dirs, files in os.walk(TEMPLATE_FOLDER):
                for file in files:
                    if file.endswith('.html'):
                        file_path = os.path.join(root, file)
                        with open(file_path, 'r') as f:
                            content = f.read().lower()  # Convert content to lowercase to ignore case
                            if search_term in content:
                                # Find lines that contain the search term
                                matching_lines = [
                                    line for line in content.splitlines() if search_term in line
                                ]
                                results.append({
                                    'file': file,
                                    'matches': matching_lines
                                })

    return render_template('search_templates.html', search_term=search_term, results=results)
AUDIO_DIR = 'static/output'
IMAGE_DIR = 'static/archived_resources'
OUTPUT_DIR = 'static/temp_exp'

@app.route('/render_avatar_sound', methods=['GET'])
def render_avatar_sound_form():
    # Get list of audio and image files
    audio_files = [f for f in os.listdir(AUDIO_DIR) if f.endswith('.mp3')]
    image_files = [f for f in os.listdir(IMAGE_DIR) if f.endswith(('.png', '.jpg'))]

    # Shuffle and limit to 40 images
    random.shuffle(image_files)
    image_files = image_files[:40]
    video ="static/temp_exp/temp_avatar.mp4"
    return render_template('mk_avatar.html', audio_files=audio_files, image_files=image_files, video=video)

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
        output_path = add_sound_to_avatar(image_path, audio_path)

        if output_path:
            # Generate a unique ID for the output file
            uid = str(uuid.uuid4())
            final_output_path = f'{OUTPUT_DIR}/{uid}_avatar.mp4'

            shutil.copy(output_path, final_output_path)
            # cp otput to the static folder static/temp_exp/temp_avatar.mp4
            shutil.copy(output_path, 'static/temp_exp/temp_avatar.mp4')
            return redirect(url_for('render_avatar_sound_form'))
            #return render_template('sound_to_avatar.html', video=final_output_path)
            #return render_template('mk_avatar.html', video=final_output_path)
        else:
            return "Error creating video 4958", 500
    else:
        return "Missing image or audio path", 400

def add_sound_to_avatar(image_path, audio_path):
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
        print(f"Error creating video 4991: {e}")
        return None

#run this '/home/jack/Desktop/Flask_Make_Art/flask_env/bin/python' 'Wav2Lip-master/makeit'
@app.route('/makeit')
def makeit():
    subprocess.run(['/home/jack/miniconda3/envs/cloned_base/bin/python', 'Wav2Lip-master/makeit'], check=True)
    return redirect(url_for('mk_videos'))

# Paths
square_folder = 'static/square'
temp_folder = 'static/temp_512x512'
video_folder = 'static/videos'
audio_file = 'static/voice/ALL_VOICE.mp3'
video_output = os.path.join(video_folder, '515x768.mp4')

# Ensure necessary directories exist
os.makedirs(temp_folder, exist_ok=True)
os.makedirs(video_folder, exist_ok=True)


# Step 1: Resize images to 512x512
def resize_square_images():
    #for filename in os.listdir(square_folder):
    for filename in os.listdir('static/square'):
        if filename.endswith(('.jpg', '.png')):
            img_path = os.path.join('static/square', filename)
            img = Image.open(img_path).convert('RGB')
            img = img.resize((512, 512), resample=Image.Resampling.LANCZOS)
            output_path = os.path.join('static/temp_512x512', os.path.splitext(filename)[0] + '.jpg')
            img.save(output_path, 'JPEG')

# Step 2: Create a long image from selected images
def create_long_image(selected_images):
    logit("Creating long image from selected images...")
    long_image_height = 0
    images = []
    # sort images alphabetically reversed
    selected_images = sorted(selected_images, reverse=True)
    # Load images and calculate total height
    for image in selected_images:
        img_path = os.path.join(temp_folder, image)
        img = Image.open(img_path)
        images.append(img)
        long_image_height += img.height

    # Create a new long image
    long_image = Image.new('RGB', (512, long_image_height))

    # Paste each image into the long image
    current_height = 0
    for img in images:
        long_image.paste(img, (0, current_height))
        current_height += img.height

    long_image_path = os.path.join(temp_folder, 'long_image.jpg')
    long_image.save(long_image_path)
    logit(f"Long image created: {long_image_path}")
    return long_image_path

# Step 3: Create scrolling video from the long image
#def create_video(long_image_path, duration=45.0):
def make_scrolling_videos(image_path, output_video_path, video_duration=45, video_size=(512, 768)):
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
    # copy the video to the static static/temp_exp/515x768X.mp4
    shutil.copy(output_video_path, 'static/temp_exp/512x768X.mp4')
    #copy with unique uuid to static/Videos
    uid = str(uuid.uuid4())
    final_output_path = f'{video_folder}/{uid}_512x768X.mp4'
    shutil.copy(output_video_path, final_output_path)
    video_path = 'static/temp_exp/512x768X.mp4'
    Video = add_halloween_title_image(video_path, hex_color = "#A52A2A")
    return Video

@app.route('/square_images')
def square_images():
    resize_square_images()
    images = os.listdir('static/temp_512x512')
    #list by date reverse
    images = sorted(images, reverse=True)
    return render_template('512x512.html', images=images)

@app.route('/create_square_video', methods=['POST'])
def create_square_video():
    selected_images = request.form.getlist('images')
    duration = request.form.get('speed', 45.0)  # Default duration is 20 seconds
    logit(f"Selected images: {selected_images}")
    logit(f"Video duration set to: {duration} seconds")

    if selected_images:
        # Create a long vertical image and then a video from it
        long_image_path = create_long_image(selected_images)
        output_video_path = os.path.join(video_folder, '515x768.mp4')
        duration = float(duration)
        image_path = long_image_path
        make_scrolling_videos(image_path, output_video_path, video_duration=duration)  # Corrected line

    return render_template('512x512.html', images=os.listdir(temp_folder), video_filename='515x768.mp4')

@app.route('/vertical_square')
def vertical_square():
    return render_template('512x512.html')

@app.route('/static/temp_512x512/<filename>')
def serve_image(filename):
    return send_from_directory(temp_folder, filename)

#square_zoomy_route subprocess bash square_zoomy4
@app.route('/square_zoomy', methods=['GET', 'POST'])
def square_zoomy_route():
    subprocess.run(['/bin/bash', 'square_zoomY4'], check=True)
    return redirect(url_for('mk_videos'))

# -----------------------------------------
import nltk
import spacy
from nltk.corpus import stopwords
from pydantic import BaseModel

# Initialize Flask app

# Load SpaCy model for NLP processing
nlp = spacy.load('en')#_core_web_sm')

class PromptModel(BaseModel):
    text: str
    min_words: int = 50
    max_words: int = 75

    def preprocess_text(self):
        """Tokenizes and filters the text, removing stopwords."""
        stop_words = set(stopwords.words('english'))
        tokens = nltk.word_tokenize(self.text)
        filtered_tokens = [word for word in tokens if word.isalnum() and word.lower() not in stop_words]
        return filtered_tokens

    def extract_key_phrases(self):
        """Extracts key phrases using SpaCy's noun chunking."""
        doc = nlp(self.text)
        key_phrases = set(chunk.text for chunk in doc.noun_chunks)
        return list(key_phrases)

    def generate_prompt(self):
        """Generates a readable prompt from key phrases."""
        phrases = self.extract_key_phrases()

        # Select some random key phrases to use in the prompt
        selected_phrases = random.sample(phrases, min(len(phrases), 5))
        prompt = ' '.join(selected_phrases)

        # Ensure prompt has at least min_words
        if len(prompt.split()) < self.min_words:
            additional_phrases = random.sample(phrases, 2)
            prompt += ' ' + ' '.join(additional_phrases)

        # Limit the prompt to max_words
        return ' '.join(prompt.split()[:self.max_words])

# Function to generate random artist prompt selections
def generate_random_prompt():
    """Generates a random artist, media, topic, style, and media prompt."""
    Views = ["Birds Eye View","High Angle shot","Eye Level shot","Low Angle shot","First-Person View",
        "Over-the-Shoulder Shot","Overhead shot","Third-Person View","Top Down View","Cinematic shot","Aerial View","Gopro view", "Tracking shot","crane shot","fish eye view","panorama 360 view","Drone shot"]
    Artist = [
        "Leonardo Da Vinci", "Rembrandt", "Vincent Van Gogh", "Claude Monet",
        "Gustav Klimt", "Paul Cezanne", "Bill Sienkiewicz", "Jean Giraud Moebius", "Caravaggio", "Francisco De Goya", "Juli Bell", "Boris Vallejo", "Frida Kahlo", "Salvador Dali","Norman Rockwell","James Abbott McNeill" ,"Bill Sienkiewicz", "Jean Giraud Moebius"
    ]
    Topic = [
        "Abstracted", "Animals", "Architecture", "Astronomy", "Ocean",
        "Mountains", "Death", "Buildings", "Forest", "Seashore", "Flowers", "desert", "beach", "river", "lake", "forest", "sea", "aliens", "wolves", "sunset", "sky", "waves", "apocalyptic textures","apocalyptic","dystopian","jungle", "insects","space" 
    ]
    Style = [
        "Abstract Art", "Steampunk", "Grunge" ,"Art Deco", "Art Nouveau", "Baroque", "Cubism", "Expressionism", "Futurism", "Impressionism", "Minimalism", "Nordic", "Pop Art", "Post-Impressionism", "Realism", "Renaissance", "Surrealism", "Symbolism", "Ukiyo-e", "Ukiyo-e", "Vaporwave"
    ]
    Media = [
        "Acrylic On Canvas", "Charcoal", "Collage", "Crayon", "Emulsion", "Enamel", "Drypoint", "Ceramic", "sketch", "chalk", "watercolor", "pastel", "acrylic", "oil", "watercolor", "ink", "pastel", "watercolor", "acrylic", "oil", "alcohol ink art"
    ]

    selection = (
        random.choice(Views),
        random.choice(Artist),
        random.choice(Media),
        random.choice(Topic),
        random.choice(Style),
        random.choice(Media)
    )

    return selection

# Function to read text from a file
def load_text_from_file(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        return file.read()

# Function to save text to a file
def save_text_to_file(text, filename):
    with open(filename, 'a', encoding='utf-8') as file:
        file.write(text + '.\n')

# Function to save the generated prompt to a file
def save_prompt(text):
    with open('prompts1.txt', 'a', encoding='utf-8') as file:
        file.write(text + '.\n')

# Route to home page for prompt generation
@app.route('/mk_prompt', methods=['GET', 'POST'])
def mk_prompt_route():
    if request.method == 'POST':
        # Save the user-submitted prompt to prompts.txt
        prompt_text = request.form['prompt_text']
        save_text_to_file(prompt_text, 'prompts.txt')
        return redirect(url_for('mk_prompt_route'))

    # Load text from prompts.txt
    TEXT = load_text_from_file('prompts.txt')
    prompt_model = PromptModel(text=TEXT)

    # Generate random artist prompts using your generator function
    generated_prompts = []
    for _ in range(3):
        prompt_generated = prompt_model.generate_prompt()
        generated_prompts.append(prompt_generated)

    # Define random top and bottom texts
    TOP = "Graphic novel cover page top text in fancy gold 3D letters: " + random.choice(
        ['"AI Generated"', '"Python Prompts"', '"Generated Prompt"', '"Flask Prompts"', '"Machine Learning Prompts"']
    )
    BOTTOM = random.choice(
        ['"Prompts by Python"', '"Prompts by AI"', '"Prompts by Machine Learning"', '"Prompts by FlaskArchitect"', '"Prompts by Flask"']
    )

    # Combine generated prompts into one text
    combined_text = TOP + ' '.join(generated_prompts) + BOTTOM
    save_prompt(combined_text + '.\n\n')

    # Generate random prompt selection for display
    random_prompt_selection = generate_random_prompt()

    # Render the template with all data
    return render_template('index_prompt_tones.html', top_text=TOP, generated_prompts=generated_prompts, bottom_text=BOTTOM, text=TEXT, random_prompt_selection=random_prompt_selection)
# -----------------------------------------
VIDEOS_PATH = '/mnt/HDD500/Image_Retriever/static/videos'
ARCHIVES_PATH = 'static/archives_results'
JSON_FILE = 'videos.json'

def extract_random_frame(video_path, output_folder, num_frames=3):
    clip = VideoFileClip(video_path)
    duration = clip.duration
    frames = []

    for i in range(num_frames):
        random_time = random.uniform(0, duration)
        frame_path = os.path.join(output_folder, f'frame_{i}.jpg')
        clip.save_frame(frame_path, t=random_time)
        frames.append(frame_path)

    return frames

def scan_videos():
    corrupt_videos = []
    
    if os.path.exists(JSON_FILE):
        with open(JSON_FILE, 'r') as f:
            return json.load(f), corrupt_videos

    videos_data = {}
    video_files = glob.glob(os.path.join(VIDEOS_PATH, '*.mp4'))

    for video in video_files:
        video_name = os.path.basename(video)
        video_folder = os.path.join(ARCHIVES_PATH, video_name)
        os.makedirs(video_folder, exist_ok=True)

        try:
            # Extract 3 random frames
            frames = extract_random_frame(video, video_folder)
            videos_data[video_name] = frames
        except Exception as e:
            print(f"Error processing video {video_name}: {e}")
            corrupt_videos.append(video_name)

    # Save results in videos.json
    with open(JSON_FILE, 'w') as f:
        json.dump(videos_data, f)

    return videos_data, corrupt_videos

@app.route('/review_videos')
def review_videos():
    videos_data, corrupt_videos = scan_videos()
    return render_template('review_videos.html', videos=videos_data, corrupt_videos=corrupt_videos)

@app.route('/delete_json', methods=['POST'])
def delete_json():
    if os.path.exists(JSON_FILE):
        os.remove(JSON_FILE)
    return redirect(url_for('index'))


def mk_temp(data):  # Create a temporary file
    with open('tempfile.txt', 'a') as inputs:
        inputs.write(data + '\n')  # Ensure each entry is on a new line

# Create the database and table if they don't exist
def create_search_database():
    conn = sqlite3.connect('history.db')
    cursor = conn.cursor()
    cursor.execute('CREATE TABLE IF NOT EXISTS history (command TEXT PRIMARY KEY)')  # Set command as PRIMARY KEY

    # Path to the Bash history file
    history_file = os.path.expanduser('~/.bash_history')

    # Read the history file and insert commands into the database
    if os.path.exists(history_file):
        with open(history_file, 'r') as file:
            for line in file:
                command = line.strip()  # Remove leading/trailing whitespace
                if command:  # Check if the line is not empty
                    try:
                        cursor.execute('INSERT INTO history (command) VALUES (?)', (command,))
                    except sqlite3.IntegrityError:
                        # This error occurs if the command already exists
                        pass

    conn.commit()
    conn.close()

# Call create_database() to initialize the database when the app starts
create_search_database()

# Function to get a database connection
def get_db_conn():
    conn = sqlite3.connect('history.db')
    conn.row_factory = sqlite3.Row
    return conn

# Load the index page
@app.route('/search_history', methods=['GET', 'POST'])
def search_history():
    history_results = []  # Initialize results for command history

    # Check for command history search
    if request.method == 'POST' and 'query' in request.form:
        query = request.form['query']
        with get_db_conn() as conn:
            cursor = conn.execute('SELECT command FROM history WHERE command LIKE ?', ('%' + query + '%',))
            history_results = cursor.fetchall()  # Fetch all matching results from the database

        # Create a temporary file for found results
        for data in history_results:
            mk_temp(data[0])  # Create a temporary file for each found result

    return render_template('search_history.html', history_results=history_results)

# Load the search for the tempfile
@app.route('/search_text_file', methods=['POST', 'GET'])
def search_text_file():
    temp_results = []  # Initialize results for tempfile.txt

    # Check for tempfile search
    if 'temp_query' in request.form:
        temp_query = request.form['temp_query']
        if os.path.exists('tempfile.txt'):
            with open('tempfile.txt', 'r') as temp_file:
                temp_results = [line.strip() for line in temp_file if temp_query in line]  # Search for temp_query in tempfile.txt

    return render_template('search_history.html', temp_results=temp_results)

# Route to remove tempfile.txt
@app.route('/remove_tempfile', methods=['POST'])
def remove_tempfile():
    if os.path.exists('tempfile.txt'):
        os.remove('tempfile.txt')  # Remove the tempfile
    return redirect(url_for('search_history'))  # Redirect back to the index page

#--------------------------------------
# Define the directory to manage files in
ALLOWED_DIRECTORIES = {
    'static': '/home/jack/Desktop/Flask_Make_Art/static/',
    'templates': '/home/jack/Desktop/Flask_Make_Art/templates/'
}


# Helper function to check allowed extensions
def allowed_file(filename):
    return os.path.splitext(filename)[1] in ALLOWED_EXTENSIONS

# Helper function to ensure valid directory access
def safe_join(base, *paths):
    target_path = os.path.join(base, *paths)
    abs_base = os.path.abspath(base)
    abs_target = os.path.abspath(target_path)

    # Ensure the target path is within the base directory
    if not abs_target.startswith(abs_base):
        raise ValueError(f"Access to {abs_target} is not allowed")
    
    return abs_target

@app.route('/file_home')
def file_home():
    """Landing page to choose directory to manage."""
    return render_template('choose_directory.html', directories=ALLOWED_DIRECTORIES.keys())

@app.route('/file_manager/<directory>')
def file_manager(directory):
    """List all files and directories within the chosen base directory."""
    if directory not in ALLOWED_DIRECTORIES:
        flash(f"Invalid directory: {directory}", "error")
        return redirect(url_for('file_home'))
    
    base_dir = ALLOWED_DIRECTORIES[directory]
    files = []
    for root, dirs, filenames in os.walk(base_dir):
        for filename in filenames:
            if allowed_file(filename):
                full_path = os.path.join(root, filename)
                relative_path = os.path.relpath(full_path, base_dir)
                files.append(relative_path)
    files.sort()
    return render_template('file_manager.html', files=files, current_directory=directory)

@app.route('/move', methods=['POST'])
def move_file():
    """Move a file to a new location."""
    directory = request.form['directory']
    base_dir = ALLOWED_DIRECTORIES[directory]

    src = safe_join(base_dir, request.form['src'])
    dest = safe_join(base_dir, request.form['dest'])

    try:
        if not os.path.isdir(os.path.dirname(dest)):
            flash("Destination directory does not exist.", "error")
            return redirect(url_for('file_manager', directory=directory))
        
        shutil.move(src, dest)
        logit(f"Moved file: {src} to {dest}")
        flash(f"Moved {os.path.basename(src)} to {dest}", "success")
    except Exception as e:
        logit(f"Failed to move file: {src} to {dest}. Error: {e}")
        flash(f"Error moving file: {e}", "error")
    
    return redirect(url_for('file_manager', directory=directory))

@app.route('/rename', methods=['POST'])
def rename_file():
    """Rename a file."""
    directory = request.form['directory']
    base_dir = ALLOWED_DIRECTORIES[directory]

    src = safe_join(base_dir, request.form['src'])
    new_name = request.form['new_name']
    dest = safe_join(base_dir, os.path.dirname(src), new_name)

    try:
        os.rename(src, dest)
        logit(f"Renamed file: {src} to {dest}")
        flash(f"Renamed {os.path.basename(src)} to {new_name}", "success")
    except Exception as e:
        logit(f"Failed to rename file: {src} to {new_name}. Error: {e}")
        flash(f"Error renaming file: {e}", "error")
    
    return redirect(url_for('file_manager', directory=directory))
# changed /delete to /deletem
# changed delete_file to /delete_filez
@app.route('/deletem', methods=['POST'])
def delete_filez():
    """Delete a file."""
    directory = request.form['directory']
    base_dir = ALLOWED_DIRECTORIES[directory]

    file_to_delete = safe_join(base_dir, request.form['file'])

    try:
        if os.path.exists(file_to_delete):
            os.remove(file_to_delete)
            logit(f"Deleted file: {file_to_delete}")
            flash(f"Deleted {os.path.basename(file_to_delete)}", "success")
        else:
            flash(f"File {os.path.basename(file_to_delete)} not found", "error")
    except Exception as e:
        logit(f"Failed to delete file: {file_to_delete}. Error: {e}")
        flash(f"Error deleting file: {e}", "error")
    
    return redirect(url_for('file_manager', directory=directory))

@app.route('/edit/<directory>/<path:filename>')
def edit_file(directory, filename):
    """Load file content for editing."""
    if directory not in ALLOWED_DIRECTORIES:
        flash(f"Invalid directory: {directory}", "error")
        return redirect(url_for('file_home'))

    base_dir = ALLOWED_DIRECTORIES[directory]
    file_path = safe_join(base_dir, filename)

    try:
        with open(file_path, 'r') as file:
            content = file.read()
        return render_template('filemanager_edit_file.html', file_content=content, filename=filename, directory=directory)
    except Exception as e:
        logit(f"Failed to load file: {file_path}. Error: {e}")
        flash(f"Error loading file: {e}", "error")
        return redirect(url_for('file_manager', directory=directory))

@app.route('/save', methods=['POST'])
def save_file():
    """Save edited file content."""
    directory = request.form['directory']
    base_dir = ALLOWED_DIRECTORIES[directory]
    filename = request.form['filename']
    file_path = safe_join(base_dir, filename)
    new_content = request.form['file_content']

    try:
        with open(file_path, 'w') as file:
            file.write(new_content)
        logit(f"File saved: {file_path}")
        flash(f"Saved changes to {filename}", "success")
    except Exception as e:
        logit(f"Failed to save file: {file_path}. Error: {e}")
        flash(f"Error saving file: {e}", "error")
    
    return redirect(url_for('file_manager', directory=directory))

@app.route('/copy', methods=['POST'])
def copy_file():
    """Copy a file to a new location."""
    directory = request.form['directory']
    base_dir = ALLOWED_DIRECTORIES[directory]

    src = safe_join(base_dir, request.form['src'])
    dest = safe_join(base_dir, request.form['dest'])

    try:
        if not os.path.isdir(os.path.dirname(dest)):
            flash("Destination directory does not exist.", "error")
            return redirect(url_for('file_manager', directory=directory))
        
        shutil.copy2(src, dest)  # shutil.copy2 to preserve metadata
        logit(f"Copied file: {src} to {dest}")
        flash(f"Copied {os.path.basename(src)} to {dest}", "success")
    except Exception as e:
        logit(f"Failed to copy file: {src} to {dest}. Error: {e}")
        flash(f"Error copying file: {e}", "error")
    
    return redirect(url_for('file_manager', directory=directory))

@app.route('/create_directory', methods=['POST'])
def create_directory():
    """Create a new directory."""
    directory = request.form['directory']
    base_dir = ALLOWED_DIRECTORIES[directory]
    new_directory = safe_join(base_dir, request.form['new_directory'])

    try:
        os.makedirs(new_directory, exist_ok=True)
        logit(f"Created directory: {new_directory}")
        flash(f"Created new directory: {new_directory}", "success")
    except Exception as e:
        logit(f"Failed to create directory: {new_directory}. Error: {e}")
        flash(f"Error creating directory: {e}", "error")
    
    return redirect(url_for('file_manager', directory=directory))
@app.route('/create_file/<directory>', methods=['GET', 'POST'])
def create_file(directory):
    # Logic to create a file in the specified directory
    # You might return a form in GET and handle the file creation in POST
    return f"Create file in {directory}"

#--------------------------------------


# Define paths for video and audio directories
VIDEO_DIR = 'static/temp_exp'
AUDIO_DIR = 'static/audio_mp3'
OUTPUT_DIR = 'static/temp_exp'

# Ensure output directory exists
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# Route to render the form (sound_to_video.html)
@app.route('/sound_to_video')
def sound_to_video():
    # Fetch available videos and audios from their respective directories
    videos = [f for f in os.listdir(VIDEO_DIR) if f.endswith('.mp4')]
    audios = [f for f in os.listdir(AUDIO_DIR) if f.endswith('.mp3')]
    
    # Render the HTML form and pass the list of videos and audios
    return render_template('soundtovideo.html', videos=videos, audios=audios)

# Route to handle combining video and audio
@app.route('/combine', methods=['POST'])
def combine_video_audio():
    # Get the video and audio file names from the form
    video = request.form.get('video')
    audio = request.form.get('audio')

    # Check if both video and audio are provided
    if not video or not audio:
        return jsonify({'error': 'No video or audio selected'}), 400

    # Define full paths to the video and audio files
    video_path = os.path.join(VIDEO_DIR, video)
    audio_path = os.path.join(AUDIO_DIR, audio)

    # Ensure the video and audio files exist
    if not os.path.exists(video_path) or not os.path.exists(audio_path):
        return jsonify({'error': 'Video or audio file not found'}), 404

    # Generate output filename
    output_filename = f"output_{video.replace('.mp4', '')}_{audio.replace('.mp3', '')}_combined.mp4"
    output_path = os.path.join(OUTPUT_DIR, output_filename)
    output_path = os.path.join(OUTPUT_DIR, "videoX.mp4")
    shutil.copy(video_path, output_path)

    try:
        # Load the video and audio clips
        video_clip = VideoFileClip(video_path)
        audio_clip = AudioFileClip(audio_path)

        # Determine the shorter duration between video and audio, then add 1 second
        min_duration = min(video_clip.duration, audio_clip.duration) + 1

        # Ensure the new duration doesn't exceed either media's total length
        min_duration = min(min_duration, video_clip.duration, audio_clip.duration)

        # Trim the video and audio to the new duration
        video_clip = video_clip.subclip(0, min_duration)
        audio_clip = audio_clip.subclip(0, min_duration)

        # Combine the trimmed video with the trimmed audio
        video_with_audio = video_clip.set_audio(audio_clip)

        # Write the final video to the output path
        video_with_audio.write_videofile(output_path, codec='libx264', audio_codec='aac')

        # Optionally copy to a temporary location for immediate access (if needed)
        temp_vid = os.path.join(VIDEO_DIR, 'text2videoX.mp4')
        shutil.copy(output_path, temp_vid)

        # Respond with success and the temporary video path
        return jsonify({'success': True, 'output_video': temp_vid})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

    finally:
        # Release resources after processing
        video_clip.close()
        audio_clip.close()


def add_sound_to_image(image_path, audio_path):
    try:
        # Log the paths for debugging
        print(f"Image Path: {image_path}")
        print(f"Audio Path: {audio_path}")
        
        # Create an ImageClip from the image
        image_clip = ImageClip(image_path)
        
        # Load the audio file
        audio_clip = AudioFileClip(audio_path)

        # Log the duration of the audio for debugging
        print(f"Audio Duration: {audio_clip.duration} seconds")
        
        # Set the duration of the image clip to match the audio clip duration
        image_clip = image_clip.set_duration(audio_clip.duration)
        
        # Set the audio to the image clip
        video_clip = image_clip.set_audio(audio_clip)
        
        # Define the output path for the video
        output_path = image_path.replace('.jpg', '_audio.mp4').replace('.png', '_audio.mp4')
        
        # Log the output path for debugging
        print(f"Output Video Path: {output_path}")
        
        # Write the final video file with an FPS (frames per second) of 24
        video_clip.write_videofile(output_path, codec='libx264', audio_codec='aac', fps=24)
        #copy to static/temp_exp/STILLX.mp4
        shutil.copy(output_path, 'static/temp_exp/STILLX.mp4')
            
        im = Image.open(image_path).convert("RGB") 
        w, h = im.size
        if w > 833:im = im.resize((512, 512), resample=Image.LANCZOS)
        if w == 832:im = im.resize((512, 768), resample=Image.LANCZOS)
        im.save("static/projects/use.jpg")
        #copy to static/projects/use.mp3
        shutil.copy(audio_path, "static/projects/use.mp3")
        subprocess.run(['/home/jack/miniconda3/envs/cloned_base/bin/python','/home/jack/Desktop/Flask_Make_Art/Wav2Lip-master/makeavatar'], check=True)
        
        return output_path
       

    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def add_sound_to_square(image_path, audio_path):
    try:
        # Log the paths for debugging
        print(f"Image Path: {image_path}")
        print(f"Audio Path: {audio_path}")
        
        # Create an ImageClip from the image
        image_clip = ImageClip(image_path)
        
        # Load the audio file
        audio_clip = AudioFileClip(audio_path)

        # Log the duration of the audio for debugging
        print(f"Audio Duration: {audio_clip.duration} seconds")
        
        # Set the duration of the image clip to match the audio clip duration
        image_clip = image_clip.set_duration(audio_clip.duration)
        
        # Set the audio to the image clip
        video_clip = image_clip.set_audio(audio_clip)
        
        # Define the output path for the video
        output_path = image_path.replace('.jpg', '_audio.mp4').replace('.png', '_audio.mp4')
        
        # Log the output path for debugging
        print(f"Output Video Path: {output_path}")
        
        # Write the final video file with an FPS (frames per second) of 24
        video_clip.write_videofile(output_path, codec='libx264', audio_codec='aac', fps=24)
        #copy to static/temp_exp/STILLX.mp4
        shutil.copy(output_path, 'static/temp_exp/halloweenX.mp4')
        vide_path = 'static/temp_exp/halloweenX.mp4'
        VIDEO = add_halloween_frame(video_path, hex_color = "#A52A2A")
        print(f"VIDEO_XXXXXXXX: {VIDEO}")
        im = Image.open(image_path).convert("RGB")
        im.save("static/projects/use.jpg")
        #copy to static/projects/use.mp3
        shutil.copy(audio_path, "static/projects/use.mp3")
        subprocess.run(['/home/jack/miniconda3/envs/cloned_base/bin/python','/home/jack/Desktop/Flask_Make_Art/Wav2Lip-master/makeavatar'], check=True)
        
        return output_path
       

    except Exception as e:
        print(f"An error occurred: {e}")
        return None
def log_memory_usage(interval=60, output_file='static/Memory_usage.info'):
    """
    Logs memory usage to a specified file at regular intervals.
    """
    while True:
        # Get current memory usage
        memory_info = psutil.virtual_memory()
        memory_usage = memory_info.used / (1024 * 1024)  # Convert to MB
        available_memory = memory_info.available / (1024 * 1024)  # Convert to MB
        
        log_data = (
            f"Timestamp: {datetime.datetime.now()} - "
            f"Flask App Memory Usage: {memory_usage:.2f} MB - "
            f"Available Memory: {available_memory:.2f} MB - "
            f"Total System Memory: {memory_info.total / (1024 * 1024):.2f} MB - "
            f"Percentage: {memory_info.percent}%\n"
        )
        
        # Log the memory usage to the file
        with open(output_file, 'a') as f:
            f.write(log_data)
        
        print(log_data)  # Optional for console output
        time.sleep(interval)

# Define the start_memory_logger function
def start_memory_logger(interval=60, output_file='static/Memory_usage.info'):
    """
    Starts the memory logger in a separate thread to avoid blocking the main Flask thread.

    Args:
    - interval (int): Time in seconds between logging memory usage.
    - output_file (str): The file to log memory usage information.
    """
    memory_thread = threading.Thread(target=log_memory_usage, args=(interval, output_file))
    memory_thread.daemon = True  # Daemonize the thread so it stops with the main process
    memory_thread.start()

# Start the memory logger when your Flask app starts
start_memory_logger(interval=60)


# Function to plot memory usage and save with a unique filename
def plot_memory_usage(log_file='static/Memory_usage.info'):
    timestamps = []
    memory_usages = []
    available_memories = []
    total_memories = []

    # Read memory usage log
    with open(log_file, 'r') as f:
        for line in f:
            if "Flask App Memory Usage" in line:
                # Example log line format:
                # "Timestamp: 2024-10-17 11:18:16.490485 - Flask App Memory Usage: 5184.33 MB - Available Memory: 6223.44 MB - Total System Memory: 11904.64 MB - Percentage: 47.7%"
                parts = line.split(' - ')
                timestamp_str = parts[0].replace("Timestamp: ", "")
                memory_usage_str = parts[1].replace("Flask App Memory Usage: ", "").replace(" MB", "")
                available_memory_str = parts[2].replace("Available Memory: ", "").replace(" MB", "")
                total_memory_str = parts[3].replace("Total System Memory: ", "").replace(" MB", "")
                
                # Parse timestamp and memory usage using existing datetime import
                timestamps.append(datetime.datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S.%f'))
                memory_usages.append(float(memory_usage_str))
                available_memories.append(float(available_memory_str))
                total_memories.append(float(total_memory_str))

    # Plot the data
    plt.figure(figsize=(5.12, 7.68))  # 512x768 size in inches (adjust dpi later)
    
    plt.plot(timestamps, memory_usages, marker='o', linestyle='-', color='b', label='Flask App Memory Usage (MB)')
    plt.plot(timestamps, available_memories, marker='o', linestyle='-', color='g', label='Available Memory (MB)')
    plt.plot(timestamps, total_memories, marker='o', linestyle='-', color='r', label='Total System Memory (MB)')
    
    plt.xlabel('Timestamp')
    plt.ylabel('Memory Usage (MB)')
    plt.title('Flask App Memory Usage Over Time')
    plt.xticks(rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    
    # Ensure the graph directory exists
    if not os.path.exists('static/graph'):
        os.makedirs('static/graph')
    
    # Generate a unique filename using current timestamp
    output_image = f"static/graph/graph_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png"

    # Save the plot as a PNG image
    os.makedirs(os.path.dirname(output_image), exist_ok=True)  # Create directory if it doesn't exist
    plt.savefig(output_image, dpi=100)  # 512x768 at 100 DPI
    plt.close()

# Call the function to generate and save a graph with a unique name
# Function to get the most recent file from the static/graph directory
def get_latest_graph():
    graph_dir = 'static/graph'
    files = [os.path.join(graph_dir, f) for f in os.listdir(graph_dir) if f.endswith('.png')]
    if not files:
        return None  # If no files exist, return None
    latest_file = max(files, key=os.path.getctime)  # Get the file with the latest creation time
    return os.path.basename(latest_file)  # Return only the filename


# Flask route to generate the graph and display it
@app.route('/memory-graph')
def memory_graph():
    # Generate a new graph before displaying
    plot_memory_usage()

    # Find the latest graph image file
    latest_graph_image = get_latest_graph()

    # If no graph is found, handle it gracefully
    if not latest_graph_image:
        return "No graph available. Please try again after memory logging."

    # Render the template with the latest graph image
    return render_template('memory_graph.html', graph_image=latest_graph_image)


# Load your Balacoon TTS model  
#tts = TTS("static/en_us_cmuartic_jets_cpu.addon") 
tts = TTS("static/en_us_hifi_jets_cpu.addon")  
   
@app.route('/balacoon', methods=['GET', 'POST'])  
def balacoon():  
    audio_files = []  
    supported_speakers = tts.get_speakers()  # Retrieve supported speakers  

    if request.method == 'POST':  
        balacoon_speaker = request.form.get('bal_speaker', str(supported_speakers))  # Default to the last speaker  
        text = request.form['text']  
        lines = text.split("\n")  

        for line in lines:  
            if line.strip():  # Ignore empty lines  
                speaker = balacoon_speaker  # Get speaker from form input  
                samples = tts.synthesize(line, speaker)  

                # Create a unique filename keep only last 25 characters of line
                line = line[-25:]
                filename = line.replace(" ", "_").replace(".", "").replace(",", "").replace("!", "") +speaker + ".wav"
                output_file_path = os.path.join(OUTPUT_DIR, filename)  

                # Save the synthesized audio to a WAV file  
                with wave.open(output_file_path, "w") as fp:  
                    fp.setparams((1, 2, tts.get_sampling_rate(), len(samples), "NONE", "NONE"))  
                    fp.writeframes(samples)  

                # Load the WAV file and increase the volume  
                audio = AudioSegment.from_wav(output_file_path)  
                louder_audio = audio + 6  # Increase volume by 6 dB  
                # Save the louder audio as an MP3 file  
                mp3_filename = "louder_" + filename.replace('.wav', '.mp3')  
                louder_audio.export(os.path.join(OUTPUT_DIR, mp3_filename), format="mp3")  
                #copy the results to static/ouput
                shutil.copy(os.path.join(OUTPUT_DIR, mp3_filename), os.path.join('static/output', mp3_filename)) 
                shutil.copy(os.path.join(OUTPUT_DIR, mp3_filename), os.path.join('static/audio_mp3', mp3_filename)) 
                audio_files.append(mp3_filename)  
        # Retrieve the supported speakers (you may want to keep this in the app's context)  
        supported_speakers = tts.get_speakers()  
        return render_template('balacoon.html', audio_files=get_balacoon_audio_files(), supported_speakers=supported_speakers,image_files=get_balacoon_image_files())  
    # Retrieve the supported speakers (you may want to keep this in the app's context)  
    supported_speakers = tts.get_speakers()  
    return render_template('balacoon.html', audio_files=get_balacoon_audio_files(), supported_speakers=supported_speakers,image_files=get_balacoon_image_files())

def get_balacoon_audio_files():  
    audio_files = glob.glob('static/output/*.mp3')
    # sort by creation date in descending order
    audio_files.sort(key=os.path.getmtime, reverse=True)
    return audio_files  

def get_balacoon_image_files():
    #image_files=glob.glob("static/novel_images/*.jpg")
    image_files=glob.glob("static/square/*.jpg")+glob.glob("static/square/*.png")
    # sort by creation date in descending order
    image_files.sort(key=os.path.getmtime, reverse=True)
    return image_files

@app.route('/balacoon_download/<filename>')  
def balacoon_download_file(filename):  
    return send_file(os.path.join(OUTPUT_DIR, filename), as_attachment=True)  

@app.route('/combine_b', methods=['POST'])
def combine_b_video_audio():
    image_file = request.form.get('image_path')
    audio_file = request.form.get('audio_path')
    print(f"Received request to combine image: {image_file} and audio: {audio_file}")

    if not image_file or not audio_file:
        print("No image or audio file selected")
        return jsonify({'error': 'No image or audio selected'}), 400

    print(f"Selected image: {image_file}")
    print(f"Selected audio: {audio_file}")

    # Construct full paths for the image and audio
    #image_path = os.path.join('static/novel_images', image_file)
    #audio_path = os.path.join('static/output', audio_file)
    image_path = image_file
    audio_path = audio_file

    print(f"Image path: {image_path}")
    print(f"Audio path: {audio_path}")
    # Generate output filename
    output_filename = f"output_{image_file.rsplit('.', 1)[0]}_{audio_file.rsplit('.', 1)[0]}.mp4"
    output_path = os.path.join('static/output', output_filename)

    try:
        print(f"Processing image: {image_path} and audio: {audio_path}")

        # Create video from image and audio
        output_path = add_sound_to_image(image_path, audio_path)
        if output_path:
            return jsonify({'success': True, 'output_video': output_path})
        else:
            return jsonify({'error': 'Error creating video 5934'}), 500

    except Exception as e:
        print(f"Error processing video and audio: {str(e)}")
        return jsonify({'error': str(e)}), 500
@app.route('/combine_h', methods=['POST'])
def combine_h_video_audio():
    image_file = request.form.get('image_path')
    audio_file = request.form.get('audio_path')
    print(f"Received request to combine image: {image_file} and audio: {audio_file}")
    print(f"XXXXXXXXXXXX: {image_file} and audio: {audio_file}")
    if not image_file or not audio_file:
        print("No image or audio file selected")
        return jsonify({'error': 'No image or audio selected'}), 400

    print(f"Selected SQUARE image: {image_file}")
    print(f"Selected SQUARE audio: {audio_file}")

    # Construct full paths for the image and audio
    image_path = os.path.join('static/square', image_file)
    audio_path = os.path.join('static/output', audio_file)


    print(f"Image square path: {image_path}")
    print(f"Audio square path: {audio_path}")
    # Generate output filename
    output_filename = f"output_{image_file.rsplit('.', 1)[0]}_{audio_file.rsplit('.', 1)[0]}.mp4"
    output_path = os.path.join('static/output', output_filename)

    try:
        print(f"Processing image: {image_path} and audio: {audio_path}")

        # Create video from image and audio
        output_path = add_sound_to_square(image_path, audio_path)
        if output_path:
            return jsonify({'success': True, 'output_video': output_path})
        else:
            return jsonify({'error': 'Error creating video 6010'}), 500

    except Exception as e:
        print(f"Error processing video and audio: {str(e)}")
        return jsonify({'error': str(e)}), 500

def size_blur_image(image_path):
    # Open the original image
    original_image = Image.open(image_path).convert("RGB")

    # Check if the image is square
    width, height = original_image.size
    if width != height:
        print(f"Skipping {image_path}: not a square image.")
        return  # Skip non-square images

    # Create a 512x768 blurred background
    blurred_background = original_image.resize((512, 768)).filter(ImageFilter.GaussianBlur(radius=15))

    # Create a 512x512 copy of the original image
    centered_image = original_image.resize((512, 512))

    # Paste the 512x512 centered image onto the blurred background
    x_offset = (512 - 512) // 2  # Horizontal centering
    y_offset = (768 - 512) // 2  # Vertical centering
    blurred_background.paste(centered_image, (x_offset, y_offset))
    image_save = 'static/square_resources/' + os.path.basename(image_path)
    # Save the final image, overwriting the original image
    if not os.path.exists('static/square_resources'):
        os.makedirs('static/square_resources')
    blurred_background.save(image_save)
    print(f"Processed and named: {image_path}")
    return image_path


@app.route('/size_blur', methods=['POST', 'GET'])
def size_blur():
    # delete 'static/square_resources/*.jpg'
    if os.path.exists('static/square_resources'):
        shutil.rmtree('static/square_resources')
    blurred = []
    image_files = glob.glob("static/square/*.jpg")+glob.glob("static/square/*.png")
    for image_path in image_files:
        image_file=size_blur_image(image_path)
        blurred.append(image_file)
    print(blurred)
    print(image_files)
    print(f"Blurred: {blurred}")
    print(f"Image files: {image_files}")
    if not len(blurred):
        return render_template('size_blurr.html', image_files=image_files)
    #return render_template('size_blurr.html', image_files=blurred)
    blurred_list = glob.glob('static/square_resources/*.jpg')
    return render_template('size_blurr.html', image_files=blurred_list)


def restore_to_square(image_path):
    # Open the original image
    original_image = Image.open(image_path).convert("RGB")

    # Check if the image is square
    width, height = original_image.size
    if width == height:
        print(f"Skipping {image_path}: square image.")
        return  # Skip square images

    # Create a 512x768 blurred background
    blurred_background = original_image.resize((512, 768)).filter(ImageFilter.GaussianBlur(radius=15))

    # Create a 512x512 copy of the original image
    centered_image = original_image.resize((512, 512))

    # Paste the 512x512 centered image onto the blurred background
    x_offset = (512 - 512) // 2  # Horizontal centering
    y_offset = (768 - 512) // 2  # Vertical centering
    blurred_background.paste(centered_image, (x_offset, y_offset))
    image_path = image_path.replace(".jpg", "_restored.jpg")
    # Save the final image, overwriting the original image
    blurred_background.save(image_path)
    if not os.path.exists('static/restored_resources'):
        os.makedirs('static/restored_resources')
    shutil.copy(image_path, 'static/restored_resources')
    print(f"Processed and named: {image_path}")
    return image_path


@app.route('/restore', methods=['POST', 'GET'])
def restore():
    # remove the current images in "static/restored_resources/
    for f in os.listdir('static/frames'):
        os.remove(os.path.join('static/frames', f))
    blurred = []
    image_files = glob.glob("static/frames/*.jpg")
    for image_path in image_files:
        image_file=restore_to_square(image_path)
        blurred.append(image_file)
    print(blurred)
    print(image_files)
    print(f"Blurred: {blurred}")
    print(f"Image files: {image_files}")
    if not len(blurred):
        return render_template('size_blurr.html', image_files=image_files)
    #return render_template('size_blurr.html', image_files=blurred)
    blurred_list = glob.glob("static/restored_resources/*_restored.jpg")
    return render_template('size_blurr.html', image_files=blurred_list)











# def convert a webp to jpg
def convert_webp_to_jpg(image_path):
    # Open the original image
    original_image = Image.open(image_path).convert("RGB")

    # Check if the image is square
    width, height = original_image.size
    if width != height:
        print(f"Skipping {image_path}: not a square image.")
        return  # Skip non-square images

    # Convert the image to JPEG format
    converted_image = original_image.convert("RGB")

def add_halloween_frame(video_path, hex_color = "#A52A2A"):
    hex_color=random.choice(["#A52A2A","#ad1f1f","#16765c","#7a4111","#9b1050","#8e215d","#2656ca"])
    # Define the directory path
    directory_path = "temp"
    # Check if the directory exists
    if not os.path.exists(directory_path):
        # If not, create it
        os.makedirs(directory_path)
        print(f"Directory '{directory_path}' created.")
    else:
        print(f"Directory '{directory_path}' already exists.") 
    # Load the video file and title image
    video_clip = VideoFileClip(video_path)
    print(video_clip.size)
    # how do i get the width and height of the video
    width, height = video_clip.size
    get_duration = video_clip.duration
    print(get_duration, width, height)
    title_image_path = "static/assets/halloween.png"
    # Set the desired size of the padded video (e.g., video width + padding, video height + padding)
    padded_size = (width + 50, height + 50)

    # Calculate the position for centering the video within the larger frame
    x_position = (padded_size[0] - video_clip.size[0]) / 2
    y_position = (padded_size[1] - video_clip.size[1]) / 2
    #hex_color = "#09723c"
    # Remove the '#' and split the hex code into R, G, and B components
    r = int(hex_color[1:3], 16)
    g = int(hex_color[3:5], 16)
    b = int(hex_color[5:7], 16)

    # Create an RGB tuple
    rgb_tuple = (r, g, b)

    # Create a blue ColorClip as the background
    blue_background = ColorClip(padded_size, color=rgb_tuple)

    # Add the video clip on top of the red background
    padded_video_clip = CompositeVideoClip([blue_background, video_clip.set_position((x_position, y_position))])
    padded_video_clip = padded_video_clip.set_duration(video_clip.duration)
    #title_image_path = "/home/jack/Desktop/EXPER/static/assets/Title_Image02.png"
    # Load the title image
    title_image = ImageClip(title_image_path)

    # Set the duration of the title image
    title_duration = video_clip.duration
    title_image = title_image.set_duration(title_duration)

    print(video_clip.size)
    # Position the title image at the center and resize it to fit the video dimensions
    #title_image = title_image.set_position(("left", "top"))
    title_image = title_image.set_position((0, -5))
    #video_clip.size = (620,620)
    title_image = title_image.resize(padded_video_clip.size)

    # Position the title image at the center and resize it to fit the video dimensions
    #title_image = title_image.set_position(("center", "center")).resize(video_clip.size)

    # Create a composite video clip with the title image overlay
    composite_clip = CompositeVideoClip([padded_video_clip, title_image])
    # Limit the length to video duration
    composite_clip = composite_clip.set_duration(video_clip.duration)
    # Load a random background music
    mp3_files = glob.glob("static/music_scary/*.mp3")
    random.shuffle(mp3_files)

    # Now choose a random MP3 file from the shuffled list
    mp_music = random.choice(mp3_files)
    get_duration = AudioFileClip(mp_music).duration
    # Load the background music without setting duration
    music_clip = AudioFileClip(mp_music)
    # Fade in and out the background music
    #music duration is same as video
    music_clip = music_clip.set_duration(video_clip.duration)
    # Fade in and out the background music
    fade_duration = 1.0
    music_clip = music_clip.audio_fadein(fade_duration).audio_fadeout(fade_duration)
    # Set the audio of the composite clip to the background music
    composite_clip = composite_clip.set_audio(music_clip)
    uid = uuid.uuid4().hex
    output_path = 'static/temp_exp/halloweenX.mp4'
    video_path = 'static/temp_exp/halloweenX.mp4'

    # Export the final video with the background music
    composite_clip.write_videofile(video_path)
    mp4_file =  f"/mnt/HDD500/collections/vids/Ready_Post_{uid}.mp4"
    shutil.copyfile(output_path, mp4_file)
    Video = overlay_text(video_path)
    logit(Video)     
    print(mp4_file)
    
    VIDEO = output_path
    return VIDEO

from moviepy.audio.fx.all import audio_fadein, audio_fadeout

def add_halloween_title_image(video_path, hex_color="#A52A2A"):

    
    # Randomly select a color from a list of Halloween-themed colors
    hex_color = random.choice(["#A52A2A", "#ad1f1f", "#16765c", "#7a4111", "#9b1050", "#8e215d", "#2656ca"])
    logit(f"Selected hex color: {hex_color}")

    # Define the directory path
    directory_path = "temp"
    
    # Check if the directory exists, if not create it
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        logit(f"Directory '{directory_path}' created.")
    else:
        logit(f"Directory '{directory_path}' already exists.") 
    
    # Load the video file
    video_clip = VideoFileClip(video_path)
    width, height = video_clip.size
    duration = video_clip.duration
    logit(f"Video size: {width}x{height}, Duration: {duration}s")
    
    # Check if the video has audio
    has_audio = video_clip.audio is not None
    if has_audio:
        logit("Video has existing audio. No external audio will be added.")
    else:
        logit("No audio detected in the video. Adding background music.")

    # Choose a random Halloween title image
    title_image_path = random.choice(glob.glob("static/assets/*halloween.png"))
    logit(f"Selected title image: {title_image_path}")

    # Define the padded size for the video with a 50px border
    padded_size = (width + 50, height + 50)

    # Calculate position to center the video within the padded area
    x_position = (padded_size[0] - width) / 2
    y_position = (padded_size[1] - height) / 2

    # Convert hex color to RGB
    r = int(hex_color[1:3], 16)
    g = int(hex_color[3:5], 16)
    b = int(hex_color[5:7], 16)
    rgb_tuple = (r, g, b)

    # Create a background color clip (as a frame) with the padded size
    background_clip = ColorClip(padded_size, color=rgb_tuple)
    logit(f"Created background clip with RGB: {rgb_tuple}")

    # Overlay the video onto the background
    padded_video_clip = CompositeVideoClip([background_clip, video_clip.set_position((x_position, y_position))])
    padded_video_clip = padded_video_clip.set_duration(duration)

    # Load the title image and resize it to match the padded video size
    title_image = ImageClip(title_image_path).set_duration(duration).resize(padded_size)
    title_image = title_image.set_position((0, -5))

    # Composite the title image over the padded video
    composite_clip = CompositeVideoClip([padded_video_clip, title_image]).set_duration(duration)

    # If the video has no audio, add background music
    if not has_audio:
        # Load a random background music track
        mp3_files = glob.glob("/mnt/HDD500/collections/music_dark/*.mp3")
        if mp3_files:
            random.shuffle(mp3_files)
            mp_music = random.choice(mp3_files)
            music_clip = AudioFileClip(mp_music).set_duration(duration)

            # Add fade in and fade out to the music
            fade_duration = 1.0
            music_clip = music_clip.audio_fadein(fade_duration).audio_fadeout(fade_duration)
            
            # Set the composite clip's audio to the music
            composite_clip = composite_clip.set_audio(music_clip)
            logit(f"Background music added: {mp_music}")
        else:
            logit("No music files found. Skipping background audio.")

    # Generate a unique ID for the output file
    uid = uuid.uuid4().hex
    output_path = 'static/temp_exp/halloweenX.mp4'
    logit(f"Output video path: {output_path}")

    # Export the final video
    composite_clip.write_videofile(output_path, codec="libx264", audio=not has_audio)
    
    # Copy the output file to the final destination
    mp4_file = f"/mnt/HDD500/collections/vids/Ready_Post_{uid}.mp4"
    shutil.copyfile(output_path, mp4_file)
    logit(f"Final video saved to: {mp4_file}")

    return mp4_file


def delete_non_X_mp4_files(directory_path):
    # Get all the .mp4 files in the directory
    mp4_files = glob.glob(os.path.join(directory_path, '*.mp4'))+glob.glob(os.path.join(directory_path, '*.wav'))+glob.glob(os.path.join(directory_path, '*.mp3'))
    
    for file_path in mp4_files:
        # Get the filename from the file path
        filename = os.path.basename(file_path)
        
        # Check if the filename ends with 'X.mp4'
        if not filename.lower().endswith('x.mp4'):
            try:
                # Delete the file if it doesn't end with 'X.mp4'
                os.remove(file_path)
                print(f"Deleted: {filename}")
            except Exception as e:
                print(f"Error deleting {filename}: {e}")
        else:
            print(f"Kept: {filename}")

# Configuration for font folder
app.config['FONT_FOLDER'] = 'static/fonts/'

@app.route('/merge_video_background', methods=['GET', 'POST'])
def merge_video_background():
    if request.method == 'POST':
        # Get the selected video and image file paths from the form
        video_filename = request.form.get('video_file')
        image_filename = request.form.get('image_file')

        video_path = os.path.join('static/temp_exp', video_filename)
        image_path = os.path.join('static/novel_images', image_filename)

        # Check if the selected video and image files exist
        if not os.path.exists(video_path) or not os.path.exists(image_path):
            return "Video or Image file not found!"

        # Load the video and image using MoviePy
        video_clip = VideoFileClip(video_path).resize((512, 512))  # Resize video to 512x312
        background_image = ImageClip(image_path).set_duration(video_clip.duration).resize((512, 768))  # Resize image to 512x768

        # Position the video in the center of the background image
        x_position = (background_image.size[0] - video_clip.size[0]) / 2  # Center horizontally
        y_position = (background_image.size[1] - video_clip.size[1]) / 2  # Center vertically

        # Create a composite video with the video on top of the image background
        composite_clip = CompositeVideoClip([background_image, video_clip.set_position((x_position, y_position))])

        # Set the output path for the intermediate video
        intermediate_output_path = "static/temp_exp/temp_merged_videoX.mp4"
        composite_clip.write_videofile(intermediate_output_path, codec="libx264")
        logit(f"Merged video saved to: {intermediate_output_path}")

        # Call overlay_text to add random Halloween-themed text to the final video
        final_video_path = overlay_text(intermediate_output_path)
        
        # Serve the video file as the response
        return send_file(final_video_path, as_attachment=True)

    # On GET request, display the HTML form
    image_files = os.listdir('static/novel_images')
    video_files = os.listdir('static/temp_exp')

    return render_template('im_on_mp4.html', image_files=image_files, video_files=video_files)

# Function to overlay random Halloween-themed text on the video
def overlay_text(video_path):
    VIDEO = video_path
    text = random.choice([
        "Spooky season vibes!", "Fright night fun!", "Witch, please!", "Trick or treat yourself!",
        "Boo-tifully spooky!", "Ghosts, ghouls, and good times!", "Witch way to the candy?",
        "Fangs for the memories!", "Eat, drink, and be scary!", "Creepin' it real!",
        "Bone-chilling fun!", "Too cute to spook!", "Ghouls just wanna have fun!",
        "This is where the magic happens!", "Spooky treats for spooky people!", 
        "Resting witch face.", "Feeling fa-boo-lous!", "It's all just a bunch of hocus pocus.",
        "Boo crew on the loose!", "Something wicked this way comes.", 
        "Monsters, witches, and bats, oh my!"
    ])
    
    # Load the video clip
    video_clip = VideoFileClip(VIDEO)
    logit(f"Loaded video: {VIDEO}, Duration: {video_clip.duration} seconds, Size: {video_clip.size}")
    
    # Set the path for the font
    font_path = os.path.join(app.config['FONT_FOLDER'], 'xkcd-script.ttf')
    
    # Create the text clip to overlay with custom position (20px from left, 50px down from top)
    text_clip = TextClip(
        text, font=font_path, fontsize=40, color="yellow", 
        stroke_color="black", stroke_width=2
    ).set_position((20, 50)).set_duration(video_clip.duration)  # Position the text 20px from the left, 50px down
    
    print(f"Selected random text: '{text}'")

    # Overlay the text clip on the video clip
    final_clip = CompositeVideoClip([video_clip, text_clip])
    
    # Generate unique filename for output
    uid = uuid.uuid4().hex
    output_path = f'static/temp_exp/halloween_textX.mp4'
    final_clip.write_videofile(output_path, codec="libx264")
    logit(f"Video with overlay saved to: {output_path}")
    
    # Define final destination for the MP4 file
    mp4_file = f"/mnt/HDD500/collections/vids/Ready_Post_{uid}.mp4"
    shutil.copyfile(output_path, mp4_file)
    logit(f"Copied final video to: {mp4_file}")
    
    return mp4_file

@app.route('/utilities', methods=['GET', 'POST'])
def utilities():
    return render_template('utilities.html')

# Define paths for uploads
UPLOAD_FOLDER = 'static/uploads'
OUTPUT_FOLDER = 'static/outputs'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER

# Ensure folders exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

@app.route('/png_overlay')
def png_overlay():
    # Serve the html form to select files and set the overlay options
    return render_template('png_overlay.html')

@app.route('/png_on_mp4', methods=['POST','GET'])
def png_on_mp4():
    # Get the uploaded files and overlay parameters
    video_file = request.files['mp4']
    png_file = request.files['png']
    start_time = float(request.form['start_time'])
    stop_time = float(request.form['stop_time'])
    x_pos = int(request.form['x_pos'])  # Get x position from form
    y_pos = int(request.form['y_pos'])  # Get y position from form

    # Save the uploaded files to the upload folder
    video_path = os.path.join('static/uploads', video_file.filename)
    png_path = os.path.join('static/uploads', png_file.filename)
    video_file.save(video_path)
    png_file.save(png_path)

    # Load the video
    video_clip = VideoFileClip(video_path)

    # Load the PNG image and set start time and duration based on start/stop times
    png_clip = ImageClip(png_path, transparent=True).set_start(start_time).set_duration(stop_time - start_time)

    # Resize the PNG if necessary (adjust as needed)
    #png_clip = png_clip.resize(height=100)

    # Set the position of the PNG overlay using the custom x and y positions
    png_clip = png_clip.set_position((x_pos, y_pos))

    # Create a composite video with the overlay
    final_clip = CompositeVideoClip([video_clip, png_clip])

    # Output video file
    output_path = os.path.join('static', f"outputp_{video_file.filename}")
    final_clip.write_videofile('static/temp_exp/png_overlayXa.mp4', codec='libx264', fps=24, audio=True)
    # run ffmpeg on generated file to verify
    subprocess.run(['ffmpeg', '-i', 'static/temp_exp/png_overlayXa.mp4', 'static/temp_exp/png_overlayX.mp4'])
    # Return the output video file to the user
    video ='static/temp_exp/png_overlayX.mp4'
    return redirect(url_for('index'))#, video=video))

@app.route('/outputp/<filename>')
def serve_output_filep(filename):
    return send_from_directory('static/outputp_', filename)
@app.route('/face_copy', methods=['GET', 'POST'])
def face_copy():
    video_dir = 'static/temp_exp'
    video_files = [f for f in os.listdir(video_dir) if f.endswith('.mp4')]
    
    return render_template('face_copy.html', video_files=video_files)

@app.route('/process_face_copy', methods=['POST'])
def process_face_copy():
    source_video = request.form['source_video']
    destination_video = request.form['destination_video']
    
    source_path = os.path.join('static/videos', source_video)
    destination_path = os.path.join('static/videos', destination_video)
    
    logit(f"Processing face copy: {source_video} to {destination_video}")
    
    # Call the face detection, lipsync, and overlay functions here
    
    # Save the final result
    result_path = os.path.join('static/projects', f'face_swap_{source_video}_to_{destination_video}.mp4')
    
    # Assuming you have the result saved
    logit(f"Result saved at: {result_path}")
    
    return f"Face copied from {source_video} to {destination_video}. <br> Saved at {result_path}"

UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


#------------------------------
@app.route('/green_screen_overlay', methods=['GET', 'POST'])
def green_screen_overlay():
    if request.method == 'POST':
        if 'background' not in request.files or 'greenscreen' not in request.files:
            return 'Missing background or greenscreen video', 400

        background_file = request.files['background']
        greenscreen_file = request.files['greenscreen']

        # Save uploaded files
        background_path = os.path.join(app.config['UPLOAD_FOLDER'], background_file.filename)
        greenscreen_path = os.path.join(app.config['UPLOAD_FOLDER'], greenscreen_file.filename)
        background_file.save(background_path)
        greenscreen_file.save(greenscreen_path)

        # Load videos
        background_clip = VideoFileClip(background_path)
        greenscreen_clip = VideoFileClip(greenscreen_path)

        # Log durations
        app.logger.info(f"Background clip duration: {background_clip.duration} seconds")
        app.logger.info(f"Greenscreen clip duration: {greenscreen_clip.duration} seconds")

        # Check if greenscreen clip exceeds background duration
        if greenscreen_clip.duration > background_clip.duration:
            app.logger.warning("Trimming greenscreen clip to match background duration.")
            greenscreen_clip = greenscreen_clip.subclip(0, background_clip.duration)

        # Check audio availability
        if background_clip.audio is not None:
            audio_duration = background_clip.audio.duration
            app.logger.info(f"Audio clip duration: {audio_duration} seconds")

            # Ensure audio duration matches video duration
            if audio_duration > background_clip.duration:
                app.logger.warning("Trimming audio clip to match background duration.")
                background_clip.audio = background_clip.audio.subclip(0, background_clip.duration)

        # Resize and position the greenscreen
        resize_width = int(request.form.get('resize_width', 500))
        resize_height = int(request.form.get('resize_height', 500))
        x_offset = int(request.form.get('x_offset', 0))
        y_offset = int(request.form.get('y_offset', 0))

        greenscreen_clip = greenscreen_clip.resize(width=resize_width, height=resize_height)
        greenscreen_clip = greenscreen_clip.set_position((x_offset, y_offset))

        # Apply the chroma key effect to make green transparent
        greenscreen_clip = make_transparent_greenscreen(greenscreen_clip)

        # Set the duration of the greenscreen clip to match the background clip
        greenscreen_clip = greenscreen_clip.set_duration(background_clip.duration)

        # Overlay greenscreen on background and set audio
        final_clip = CompositeVideoClip([background_clip, greenscreen_clip])

        # If the background has audio, set it to the final clip
        if background_clip.audio is not None:
            final_clip = final_clip.set_audio(background_clip.audio)

        # Save the output video
        output_path = os.path.join(app.config['UPLOAD_FOLDER'], 'output.mp4')
        try:
            final_clip.write_videofile(output_path, codec='libx264', audio_codec='aac')
        except OSError as e:
            app.logger.error(f"Error during video processing: {e}")
            return str(e), 500

        # Send the output video back to the client
        return send_from_directory(app.config['UPLOAD_FOLDER'], 'output.mp4', as_attachment=True)

    return render_template('green_screen.html')

def make_transparent_greenscreen(clip):
    """Apply a chroma key effect to make green pixels transparent."""
    def remove_green(get_frame, t):
        frame = get_frame(t)

        # Create a mask based on green color
        mask = (frame[:, :, 0] < 100) & (frame[:, :, 1] > 150) & (frame[:, :, 2] < 100)

        # Set green pixels to transparent (RGBA format)
        frame = np.dstack((frame, np.ones((frame.shape[0], frame.shape[1]), dtype=frame.dtype) * 255))  # Add alpha channel
        frame[mask] = [0, 0, 0, 0]  # Set the green pixels to transparent

        return frame

    return clip.fl(remove_green)

UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


# Ensure the upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/green_screen', methods=['GET', 'POST'])
def greenscreen():
    if request.method == 'POST':
        greenscreen_file = request.files.get('greenscreen')
        background_file = request.files.get('background')

        # Get resize and offset parameters
        resize_width = int(request.form.get('resize_width', 300))
        resize_height = int(request.form.get('resize_height', 300))
        x_offset = int(request.form.get('x_offset', 0))
        y_offset = int(request.form.get('y_offset', 0))

        if greenscreen_file and background_file:
            greenscreen_path = os.path.join(app.config['UPLOAD_FOLDER'], greenscreen_file.filename)
            background_path = os.path.join(app.config['UPLOAD_FOLDER'], background_file.filename)

            # Save uploaded files
            greenscreen_file.save(greenscreen_path)
            background_file.save(background_path)

            output_video_path = os.path.join(UPLOAD_FOLDER, 'output_video.mp4')
            process_g_video(greenscreen_path, background_path, output_video_path, resize_width, resize_height, x_offset, y_offset)

            return render_template('green_screen2.html', output_video=output_video_path)

    return render_template('green_screen2.html')

def process_g_video(greenscreen_path, background_path, output_path, resize_width, resize_height, x_offset, y_offset):
    try:
        # Load the greenscreen video
        greenscreen_clip = VideoFileClip(greenscreen_path)
        audio_clip = greenscreen_clip.audio  # Extract audio from greenscreen video

        # Load the background video
        background_clip = VideoFileClip(background_path)

        # Resize the greenscreen video
        greenscreen_clip = greenscreen_clip.resize(newsize=(resize_width, resize_height))

        # Create a mask for the greenscreen
        greenscreen_mask = create_green_mask(greenscreen_clip)

        # Apply the mask to the greenscreen clip
        greenscreen_clip = greenscreen_clip.set_mask(greenscreen_mask)

        # Composite the videos
        final_clip = CompositeVideoClip([background_clip, greenscreen_clip.set_position((x_offset, y_offset))])

        # Combine audio
        combined_audio = create_combined_audio(audio_clip, background_clip.audio)
        final_clip = final_clip.set_audio(combined_audio)

        # Write the output video
        final_clip.write_videofile(output_path, codec='libx264', audio_codec='aac')

        logit("Video processing complete, saved to: {}".format(output_path))

    except Exception as e:
        error_message = f"Error processing video: {str(e)}\nTraceback: {traceback.format_exc()}"
        logit(error_message)  # Log the full error with traceback

def create_combined_audio(greenscreen_audio, background_audio):
    if background_audio is not None:
        # Start background audio after greenscreen audio
        background_audio = background_audio.set_start(greenscreen_audio.duration)
        return concatenate_audioclips([greenscreen_audio, background_audio])
    return greenscreen_audio

def create_green_mask(clip):
    """
    Create a mask for the greenscreen video by isolating green colors.
    This returns a valid MoviePy mask.
    """
    def make_mask(get_frame, t):
        frame = get_frame(t)
        # Convert the frame to HSV color space
        hsv_frame = cv2.cvtColor(frame[..., :3], cv2.COLOR_RGB2HSV)

        # Define the range for the green color in HSV
        lower_green = np.array([40, 100, 100])  # Lower bound for green
        upper_green = np.array([80, 255, 255])  # Upper bound for green

        # Create a mask where green is present
        mask = cv2.inRange(hsv_frame, lower_green, upper_green)

        # Log the sum of mask pixels to check the number of detected greens
        logit(f"Mask sum at time {t}: {np.sum(mask)}")  # Log sum of mask pixels

        # Normalize to 0-1 range and ensure it's a float32 mask
        mask = mask.astype(np.float32) / 255.0
        
        # Invert mask: green becomes transparent, non-green becomes visible
        mask = 1 - mask
        
        # Return a valid MoviePy mask by wrapping the numpy array in a VideoClip
        return mask[..., np.newaxis]  # Add a channel dimension

    # Create a VideoClip for the mask
    mask_clip = clip.fl(make_mask)
    
    # Set the duration of the mask clip to match the original clip
    mask_clip = mask_clip.set_duration(clip.duration)

    # Mark this clip as a mask
    mask_clip.ismask = True  # Explicitly indicate that this clip is a mask

    return mask_clip

VIDEO_DIR = 'static/temp_exp'
IMAGE_DIR = 'static/novel_images'

# Default JSON data for animation if none is provided
default_animation_data = [
    {"zoom": 1.0, "pan": [0, 0]},
    {"zoom": 1.0, "pan": [0, 0]},
    {"zoom": 1.2, "pan": [0.1, 0.1]},
    {"zoom": 1.5, "pan": [0.2, 0.2]},
    {"zoom": 2.0, "pan": [0.4, 0.4]},
    {"zoom": 2.0, "pan": [0.4, 0.4]},
    {"zoom": 2.0, "pan": [0.4, 0.4]},
    {"zoom": 1.2, "pan": [0.1, 0.8]},
    {"zoom": 1.2, "pan": [0.1, 0.8]},
    {"zoom": 3.2, "pan": [0.1, 0.8]},
    {"zoom": 3.2, "pan": [0.1, 0.8]},
    {"zoom": 1.2, "pan": [0.1, 0.8]},
    {"zoom": 1.2, "pan": [0.1, 0.8]},
    {"zoom": 2.5, "pan": [0.6, 0.6]},
    {"zoom": 3.0, "pan": [0.8, 0.8]},
    {"zoom": 2.5, "pan": [0.6, 0.6]},
    {"zoom": 1.5, "pan": [0.2, 0.2]},
    {"zoom": 1.0, "pan": [0, 0]}
]

@app.route('/jsonzoom')
def jsonzoom():
    # Retrieve image filenames from static/novel_images
    image_folder = 'static/novel_images'
    images = [file for file in os.listdir(image_folder) if file.endswith(('.jpg', '.jpeg', '.png'))]

    # Pass default animation data as JSON
    return render_template('json_zoom.html', 
                           images=images, 
                           default_animation_data=json.dumps(default_animation_data))


@app.route('/create_animation', methods=['POST'])
def create_animation():
    try:
        data = request.get_json()
        image_name = data.get('image')
        animation_data = data.get('animation_data', [])

        if not isinstance(animation_data, list) or len(animation_data) == 0:
            return jsonify({"error": "Invalid animation data."}), 400

        # Paths
        image_path = os.path.join('static', 'novel_images', image_name)
        output_video_path = os.path.join("static/temp_exp", 'output.mp4')

        # Create the zoom and pan animation
        create_zoom_pan_animation(image_path, animation_data, output_video_path)

        return jsonify({"message": "Video created successfully", "video_path": url_for('static', filename='output.mp4')})
    
    except Exception as e:
        print(f"Error in create_animation: {str(e)}")
        return jsonify({"error": "An error occurred while creating the animation."}), 500

def create_zoom_pan_animation(image_path, animation_data, output_video_path, interpolation_steps=60, frame_rate=10.0):
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Unable to load image at {image_path}")
        return

    # Get image dimensions
    original_height, original_width = image.shape[:2]

    # Set the output frame dimensions
    output_width, output_height = 512, 768

    # Video writer for mp4 format
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_video_path, fourcc, frame_rate, (output_width, output_height))

    # Loop through keyframes and interpolate between them
    frames = []
    for i in range(len(animation_data) - 1):
        start_frame = animation_data[i]
        end_frame = animation_data[i + 1]

        # Interpolate between start and end frames
        for step in range(interpolation_steps):
            zoom = start_frame["zoom"] + (end_frame["zoom"] - start_frame["zoom"]) * (step / interpolation_steps)
            pan_x = start_frame["pan"][0] + (end_frame["pan"][0] - start_frame["pan"][0]) * (step / interpolation_steps)
            pan_y = start_frame["pan"][1] + (end_frame["pan"][1] - start_frame["pan"][1]) * (step / interpolation_steps)

            crop_width = int(original_width / zoom)
            crop_height = int(original_height / zoom)

            x_offset = int(pan_x * (original_width - crop_width))
            y_offset = int(pan_y * (original_height - crop_height))

            x_offset = max(0, min(x_offset, original_width - crop_width))
            y_offset = max(0, min(y_offset, original_height - crop_height))

            cropped_image = image[y_offset:y_offset + crop_height, x_offset:x_offset + crop_width]
            resized_image = cv2.resize(cropped_image, (output_width, output_height))

            frames.append(resized_image)

    for frame in frames:
        video_writer.write(frame)

    video_writer.release()
    #static/temp_exp/sharpened_output.mp4
    print(f"Video saved successfully at {output_video_path}")
    subprocess.call(['ffmpeg', '-i', output_video_path, '-vf', 'unsharp=5:5:1.0:5:5:0.0', '-filter:v', 'setpts=.2*PTS', '-y', 'static/temp_exp/sharpened_output.mp4'])
    output_video = 'static/temp_exp/stable_output.mp4'
    stabilize_video('static/temp_exp/sharpened_output.mp4', output_video)
    final_video_path = os.path.join('static/temp_exp', 'final_zoomyyX.mp4')
    subprocess.call([
            'ffmpeg', '-hide_banner', '-i', 'static/temp_exp/stable_output.mp4', '-i', 'static/assets/zoom_pan_json_border_l.png',
            '-filter_complex', 
            '[0:v]scale=w=512:h=748:force_original_aspect_ratio=decrease,pad=512:768:(512-iw)/2:(778-ih)/2[v0];'
            '[1:v][v0]overlay=15:36', 
            '-codec:a', 'copy', '-y', final_video_path
        ])
    return redirect(url_for('jsonzoom'))




def stabilize_video(input_video, output_video):
    logit(f"Stabilizing video: {input_video}")
    subprocess.call([
        'ffmpeg', '-i', input_video, '-vf', 
        'vidstabdetect=shakiness=5:accuracy=10', '-f', 'null', '-'
    ])
    subprocess.call([
        'ffmpeg', '-i', input_video, '-vf', 
        'vidstabtransform=smoothing=5:input=transforms.trf', '-y', output_video
    ])
    logit(f"Stabilized video saved at: {output_video}")

@app.route('/json_zoom', methods=['GET', 'POST'])
def json_zoom():
    # Retrieve image filenames from static/novel_images
    image_folder = 'static/novel_images'
    images = [file for file in os.listdir(image_folder) if file.endswith(('.jpg', '.jpeg', '.png'))]

    output_video_path = os.path.join(VIDEO_DIR, 'json_zoomX.mp4')
    create_zoom_pan_animation(selected_image, default_animation_data, output_video_path, interpolation_steps=120, frame_rate=15.0)

    sharpened_output = os.path.join(VIDEO_DIR, 'sharpened_output.mp4')
    subprocess.call(['ffmpeg', '-i', output_video_path, '-vf', 'unsharp=5:5:1.0:5:5:0.0', '-filter:v', 'setpts=.2*PTS', '-y', sharpened_output])
    stabilize_video(sharpened_output, os.path.join(VIDEO_DIR, 'stabilized_output.mp4'))

    final_video_path = os.path.join(VIDEO_DIR, 'final_zoomyyX.mp4')
    subprocess.call([
            'ffmpeg', '-hide_banner', '-i', sharpened_output, '-i', 'static/assets/zoom_pan_json_border_l.png',
            '-filter_complex', 
            '[0:v]scale=w=512:h=748:force_original_aspect_ratio=decrease,pad=512:768:(512-iw)/2:(778-ih)/2[v0];'
            '[1:v][v0]overlay=15:36', 
            '-codec:a', 'copy', '-y', final_video_path
        ])
    logit(f"Final video created at: {final_video_path}")

                # Pass default animation data as JSON
    return render_template('json_zoom.html', 
                           images=images, 
                           default_animation_data=json.dumps(default_animation_data),video_file='temp_exp/final_zoomyyX.mp4',)
        

    # Handle GET request, render the selection page
    images = os.listdir(IMAGE_DIR)
    return render_template('json_zoom.html', images=images)
@app.route('/tinymce', methods=['GET', 'POST'])
def tinymce():
    return render_template('tinymce.html')

def novel_image_paths():
    image_paths = glob.glob(os.path.join(app.config['NOVEL_IMAGES'], '*.png'))+glob.glob(os.path.join(app.config['NOVEL_IMAGES'], '*.jpg'))
    return image_paths

@app.route('/mk_novel', methods=['GET', 'POST'])    
#@cache.cached(timeout=60)
def mk_novel():
    session['visited_index'] = True
    image_paths = novel_image_paths()
    return render_template('mk_novel.html', image_paths=image_paths)

import os
from moviepy.editor import ImageClip, concatenate_videoclips, CompositeVideoClip,VideoFileClip, ColorClip, AudioFileClip
from logit import *
import glob
import random
import uuid
import shutil
import inspect



# Directory containing the images
images_dir = 'static/novel_images/'
output_video = 'static/temp_exp/novel_video.mp4'

# Frame size and timing parameters
frame_width, frame_height = 720,1424
static_duration = 7
slide_duration = .5


def create_image_frame(image1, image2):
    """Stack two images vertically to create a 1024x2048 frame."""
    clip1 = ImageClip(image1).resize((frame_width, frame_width))
    clip2 = ImageClip(image2).resize((frame_width, frame_width))
    stacked_frame = CompositeVideoClip([clip1.set_position(("center", "top")), clip2.set_position(("center", "bottom"))],
                                       size=(frame_width, frame_height))
    return stacked_frame.set_duration(static_duration)

def create_sliding_transition(frames):
    """Create a sliding transition effect with delay for each frame pair."""
    video_clips = []
    for i in range(len(frames) - 1):
        # Current static frame
        current_frame = frames[i]
        
        # Next frame with slide-in effect
        next_frame = frames[i+1].set_start(static_duration).set_duration(slide_duration)

        # Composite to show slide-up effect
        transition_clip = CompositeVideoClip([
            current_frame.set_position(("center", "center")).set_duration(static_duration + slide_duration),
            next_frame.set_position(lambda t: ("center", frame_height * (1 - t / slide_duration) - frame_height))  # slide in
        ], size=(frame_width, frame_height)).set_duration(static_duration + slide_duration)
        
        # Add transition clip to the video sequence
        video_clips.append(transition_clip)
    
    # Add the last frame without transition
    video_clips.append(frames[-1].set_duration(static_duration))
    
    return concatenate_videoclips(video_clips, method="compose")
@app.route('/create_novel_video', methods=['POST', 'GET'])
def create_novel_video():
    images = load_novel_images(images_dir)
    
    # Ensure an even number of images for pairing
    if len(images) % 2 != 0:
        images = images[:-1]

    # Group images into pairs and create 2-image frames
    frames = []
    for i in range(0, len(images), 2):
        frame = create_image_frame(images[i], images[i+1])
        frames.append(frame)

    # Apply sliding transition between each frame pair
    final_video = create_sliding_transition(frames)
    
    # Export the final video
    final_video.write_videofile(output_video, fps=24)
    video_path = output_video
    add_title(video_path, hex_color="#A52A2A")
    return render_template('mk_novel_2.html', image_paths=novel_image_paths(), video_path=video_path)


@app.route('/preview_novel_video', methods=['POST', 'GET'])
def preview_novel_video():
    images = load_novel_images(images_dir)
    
    video_path = output_video
    
    return render_template('mk_novel_2.html', images=novel_image_paths(), video_path=video_path)

def add_title(video_path, hex_color="#A52A2A"):
    hex_color = random.choice(["#A52A2A", "#ad1f1f", "#16765c", "#7a4111", "#9b1050", "#8e215d", "#2656ca"])

    # Load the video file
    video_clip = VideoFileClip(video_path)
    width, height = video_clip.size
    video_duration = video_clip.duration

    # Set the desired size of the padded video (extra padding for title)
    padded_size = (width + 50, height + 50)
    title_image_path = "static/assets/720x1424.png"
    
    # Calculate the position for centering the video within the larger frame
    x_position = (padded_size[0] - width) / 2
    y_position = (padded_size[1] - height) / 2

    # Convert hex color to RGB
    r = int(hex_color[1:3], 16)
    g = int(hex_color[3:5], 16)
    b = int(hex_color[5:7], 16)
    rgb_tuple = (r, g, b)

    # Create a background color clip
    background_clip = ColorClip(padded_size, color=rgb_tuple)

    # Add the video clip on top of the background
    padded_video_clip = CompositeVideoClip([background_clip, video_clip.set_position((x_position, y_position))])
    padded_video_clip = padded_video_clip.set_duration(video_duration)

    # Load the title image and set its duration to match the video
    title_image = ImageClip(title_image_path).set_duration(video_duration)

    # Resize the title image to cover the entire video frame
    title_image = title_image.resize(padded_video_clip.size)

    # Position the title image over the entire frame (centered by default)
    title_image = title_image.set_position(("center", "center"))

    # Combine the padded video and the title image
    composite_clip = CompositeVideoClip([padded_video_clip, title_image])
    composite_clip = composite_clip.set_duration(video_duration)

    # Load a random background music
    mp3_files = glob.glob("/mnt/HDD500/collections/music_long/*.mp3")
    random.shuffle(mp3_files)
    mp_music = random.choice(mp3_files)
    music_clip = AudioFileClip(mp_music).set_duration(video_duration)

    # Set the audio of the composite clip to the background music
    composite_clip = composite_clip.set_audio(music_clip)

    # Save the final video
    uid = uuid.uuid4().hex
    output_path = f'static/temp_exp/novel_creationX.mp4'
    composite_clip.write_videofile(output_path)
    shutil.copyfile(output_path, f'/mnt/HDD500/collections/vids/square_mask_transitionX_{uid}.mp4')
    #get basename of output_path
    bas = os.path.basename(output_path)
    video = bas
    return render_template('mk_novel_2.html', video = bas)
#------------------------------------

app.config['DATABASE'] = 'static/blog3.db'
#DATABASE = 'static/blog3.db'
app.config['DATABASE'] = 'static/blog4.db'
DATABASE = 'static/blog4.db'


#def allowed_file(filename):
def allowed_file(filename):
    logit(f"XXXXX__filename: {filename}")
    logit(f"filename.rsplit: {filename.rsplit('.', 1)}")
    logit(f"'.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']")
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/upload_novel_video/<int:post_id>', methods=['POST'])
def upload_novel_video(post_id):
    if 'videoFile' not in request.files:
        print('No file part')
        flash('No file part')
        return redirect(request.url)
    
    file = request.files['videoFile']
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['VUPLOAD_FOLDER'], filename))
        print(f"Filename: {filename}")
        # Update the database with the filename
        update_video_filename(post_id, filename)
        flash('Video uploaded successfully')
        return redirect(url_for('post', post_id=post_id))
    else:
        flash('Allowed file types are .mp4')
        return redirect(request.url)




def list_files_by_creation_time(file_paths):
    """
    List files by their creation time, oldest first.
    Args:
    file_paths (list): List of file paths.
    Returns:
    list: List of file paths sorted by creation time.
    """
    # Log the start of the function
    print('Listing files by creation time...')
    # Create a dictionary to store file paths and their creation times
    file_creation_times = {}
    # Iterate through each file path
    for file_path in file_paths:
        # Get the creation time of the file
        try:
            creation_time = os.path.getctime(file_path)
            # Store the file path and its creation time in the dictionary
            file_creation_times[file_path] = creation_time
        except FileNotFoundError:
            # Log a warning if the file is not found
            print(f'File not found: {file_path}')
    # Sort the dictionary by creation time
    sorted_files = sorted(file_creation_times.items(), key=lambda x: x[1],reverse=True)
    # Extract the file paths from the sorted list
    sorted_file_paths = [file_path for file_path, _ in sorted_files]
    # Log the end of the function
    print('File listing complete.')
    # Return the sorted file paths
    return sorted_file_paths

def read_text_from_file(filename):
    filepath = os.path.join(TEXT_FILES_DIR, filename)
    with open(filepath, "r") as file:
        text = file.read()
        print(f"Text read from file: {filename}")
        return text
 #---------------------------- 

def load_novel_images(novel_images_dir):
    """Load and sort images from the directory."""

    images = sorted([os.path.join(images_dir, f) for f in os.listdir(images_dir) if f.endswith('.jpg')])

    return images

@app.route("/start_project", methods=["GET", "POST"])
def start_project():
    images=[]
    novel_images_dir = glob.glob("static/novel_images/*.jpg")+glob.glob("static/novel_images/*.png")
    # sort the images by creation time (reversed)
    novel_images_dir = sorted(novel_images_dir, key=os.path.getctime, reverse=True)
    for image in novel_images_dir:
        images.append(os.path.basename(image))
    return render_template("start_project2.html", images=images)

#-----------------------------    

def get_post(post_id):
    with sqlite3.connect(DATABASE) as conn:
        cursor = conn.cursor()
        cursor.execute('SELECT id, title, content, image, video_filename FROM post WHERE id = ? ORDER BY id DESC', (post_id,))
        post = cursor.fetchone()
    return post
# Function to fetch all posts
def get_posts(limit=None):
    with sqlite3.connect(DATABASE) as conn:
        cursor = conn.cursor()
        if limit:
            cursor.execute('SELECT id, title, content, image, video_filename FROM post ORDER BY id DESC LIMIT ?', (limit,))
        else:
            cursor.execute('SELECT id, title, content, image, video_filename FROM post ORDER BY id DESC')
        posts = cursor.fetchall()
    return posts
def get_intro(limit=1):
    print(f"Fetching post with limit={limit}")
    with sqlite3.connect(DATABASE) as conn:
        cursor = conn.cursor()
        try:
            if limit:
                print("Fetching post with id = 864")
                cursor.execute('SELECT id, title, content, image, video_filename FROM post WHERE id = ?', (864,))
            else:
                print("Fetching all posts ordered by id DESC")
                cursor.execute('SELECT id, title, content, image, video_filename FROM post ORDER BY id DESC')
                
            posts = cursor.fetchall()
            
            return posts
        except sqlite3.OperationalError as e:
            print(f"SQLite error occurred: {e}")
            raise
# Function to fetch image data
def get_image(post_id):
    with sqlite3.connect(DATABASE) as conn:
        cursor = conn.cursor()
        cursor.execute('SELECT image FROM post WHERE id = ?', (post_id,))
        post = cursor.fetchone()
        if post and post[0]:
            return post[0]  # Return base64 encoded image data
        return None
#-----------------------------
from moviepy.video.compositing.transitions import slide_in
from moviepy.video.fx import all
from moviepy.editor import *


# Create the output directory if it doesn't exist
output_directory = 'static/temp'
os.makedirs(output_directory, exist_ok=True)

@app.route("/mk_flipnovel", methods=["POST", "GET"])
def mk_flipnovel():
    # Set up paths
    output_directory = 'static/temp'
    final_output_directory = "/home/jack/Desktop/HDD500/collections/vids/"
    os.makedirs(output_directory, exist_ok=True)
    os.makedirs(final_output_directory, exist_ok=True)

    # Clear temporary files
    for f in glob.glob(f"{output_directory}/*"):
        os.remove(f)

    # Load images from the directory by creation time reversed
    novel_images_dir = glob.glob("static/novel_images/*.jpg")+glob.glob("static/novel_images/*.png")
    image_list = sorted(novel_images_dir, key=os.path.getctime, reverse=True)
    if not image_list:
        return "Error: No images found in the input directory."

    video_paths = []

    # Generate video segments
    for i, base_image_path in enumerate(image_list[:-1]):
        try:
            base_image = Image.open(base_image_path).convert("RGBA")
            next_image_path = image_list[i + 1]
            next_image = Image.open(next_image_path).convert("RGBA")

            # Generate frames for the transition
            frames = []
            bs = (500, 766)  # Base size for resizing
            step = 10  # Step size for transition
            hold_frames = 30  # Frames to hold before transition
            delay_frames = 120  # Frames to hold after transition

            # Hold on the current image
            for _ in range(hold_frames):
                frames.append(np.array(base_image))

            # Create the flip transition
            for j in range(0, bs[0], step):
                current_frame = next_image.copy()
                resized_base_image = base_image.resize((bs[0] - j, bs[1]), Image.BICUBIC)
                current_frame.paste(resized_base_image, (0, 0), resized_base_image)
                frames.append(np.array(current_frame))

            # Hold on the final frame
            for _ in range(delay_frames):
                frames.append(np.array(current_frame))

            # Save the video segment
            output_video_path = os.path.join(output_directory, f"segment_{i}.mp4")
            imageio.mimsave(output_video_path, frames, fps=30)
            video_paths.append(output_video_path)
        except Exception as e:
            print(f"Error creating video for {base_image_path}: {e}")
            continue

    # Ensure at least one video segment was created
    if not video_paths:
        return "Error: No video segments were created."

    # Generate ffmpeg input list
    input_list_path = os.path.join(output_directory, "input_list.txt")
    try:
        with open(input_list_path, "w") as f:
            for video_path in video_paths:
                f.write(f"file '{os.path.abspath(video_path)}'\n")
    except Exception as e:
        return f"Error: Unable to create input list - {e}"

    # Concatenate video segments
    concatenated_video_path = os.path.join(output_directory, "final_novel_result.mp4")
    ffmpeg_command = f"ffmpeg -y -f concat -safe 0 -i {input_list_path} -c copy {concatenated_video_path}"
    try:
        subprocess.run(ffmpeg_command, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        return f"Error: ffmpeg concatenation failed - {e}"

    if not os.path.exists(concatenated_video_path):
        return "Error: Concatenation failed."

    # Move the final video
    final_output_video_path = os.path.join(final_output_directory, f"{uuid.uuid4()}.mp4")
    shutil.copyfile(concatenated_video_path, final_output_video_path)

    # Add a title and return
    try:
        add_flip_title(final_output_video_path, hex_color="#A52A2A")
    except Exception as e:
        return f"Error adding title: {e}"

    return redirect("/start_project")


def add_flip_title(video_path, hex_color="#A52A2A"):
    import os
    import random
    from moviepy.editor import (
        VideoFileClip,
        CompositeVideoClip,
        ColorClip,
        ImageClip,
        AudioFileClip,
    )
    
    directory_path = "static/temp"
    os.makedirs(directory_path, exist_ok=True)

    # Randomize background color
    hex_color = random.choice(["#A52A2A", "#ad1f1f", "#16765c", "#7a4111", "#9b1050", "#8e215d", "#2656ca"])
    r, g, b = tuple(int(hex_color[i:i + 2], 16) for i in (1, 3, 5))

    # Load the video clip
    video_clip = VideoFileClip(video_path)
    video_duration = video_clip.duration  # Use video duration to align music and video

    # Load title image
    title_image_path = "/mnt/HDD500/collections/assets/graphic_novel_title.png"
    title_image = ImageClip(title_image_path).resize(video_clip.size).set_duration(video_duration)

    # Create a colored background
    background = ColorClip(video_clip.size, color=(r, g, b)).set_duration(video_duration)

    # Composite the video with the background and title
    composite = CompositeVideoClip([background, video_clip.set_position("center"), title_image.set_position("center")])

    # Load and set a random background music
    mp3_files = glob.glob("/mnt/HDD500/collections/music_long/*.mp3")
    if not mp3_files:
        raise FileNotFoundError("No MP3 files found in the specified directory.")
    random.shuffle(mp3_files)
    mp_music = random.choice(mp3_files)
    music_clip = AudioFileClip(mp_music).subclip(0, min(video_duration, AudioFileClip(mp_music).duration))

    # Fade in and fade out the music
    fade_duration = 1.0  # 1-second fade in/out
    music_clip = music_clip.audio_fadein(fade_duration).audio_fadeout(fade_duration)

    # Set the audio of the composite clip to the background music
    composite = composite.set_audio(music_clip)

    # Save the final video
    output_path = os.path.join("static/temp_exp", f"final_flipbookX.mp4")
    composite.write_videofile(output_path, codec="libx264", audio_codec="aac")

    return output_path

app.config['UPLOAD_NOVEL_IMAGES'] = 'static/temp_exp'
app.config['OUTPUT_NOVEL_IMAGES'] = 'static/edited_videos'
app.config['OUTPUT_NOVEL_DIR'] = 'static/novel_audio_mp3'
os.makedirs(app.config['OUTPUT_NOVEL_IMAGES'], exist_ok=True)
os.makedirs(app.config['OUTPUT_NOVEL_DIR'], exist_ok=True)




@app.route("/novel_audio", methods=["POST", "GET"])
def novel_audio():
    # List available videos in the temp_exp NOVEL_IMAGES
    videos = sorted(
    [f for f in os.listdir(app.config['UPLOAD_NOVEL_IMAGES']) if f.endswith('.mp4')],
    key=lambda x: os.path.getmtime(os.path.join(app.config['UPLOAD_NOVEL_IMAGES'], x)),reverse=True)
    return render_template("add_novel_audio.html", videos=videos)

@app.route("/get_video/<filename>")
def get_video(filename):
    # Serve a selected video file
    return send_from_directory(app.config['UPLOAD_NOVEL_IMAGES'], filename)

@app.route("/add_novel_audio", methods=["POST"])
def add_novel_audio():
    try:
        video_file = request.form["video_file"]
        audio_file = request.form["audio_file"]
        start_time = float(request.form["start_time"])

        video_path = os.path.join(app.config['UPLOAD_NOVEL_IMAGES'], video_file)
        audio_path = os.path.join(app.config['UPLOAD_NOVEL_IMAGES'], audio_file)

        # Create non-destructive output file path
        output_file = f"{uuid.uuid4()}_preview.mp4"
        output_path = os.path.join(app.config['OUTPUT_NOVEL_IMAGES'], output_file)

        # Load the video and audio, and set the audio timing
        video = VideoFileClip(video_path)
        audio = AudioFileClip(audio_path).set_start(start_time)
        video_with_audio = CompositeVideoClip([video.set_audio(audio)])

        # Write the new video with added audio to the output folder
        video_with_audio.write_videofile(output_path, codec="libx264", audio_codec="aac")

        return jsonify({"output_url": url_for("get_video", filename=output_file)})
    except Exception as e:
        return jsonify({"error": str(e)})
#tts = TTS("static/en_us_cmuartic_jets_cpu.addon") 
tts = TTS("static/en_us_hifi_jets_cpu.addon")
#if 'static/novel_audio_mp3' create it 

   
@app.route('/novel_balacoon', methods=['GET', 'POST'])  
def novel_balacoon():  
    audio_files = []  
    supported_speakers = tts.get_speakers()  # Retrieve supported speakers  

    if request.method == 'POST':  
        balacoon_speaker = request.form.get('bal_speaker', str(supported_speakers))  # Default to the last speaker  
        text = request.form['text']  
        lines = text.split("\n")  

        for line in lines:  
            if line.strip():  # Ignore empty lines  
                speaker = balacoon_speaker  # Get speaker from form input  
                samples = tts.synthesize(line, speaker)  

                # Create a unique filename keep only last 25 characters of line
                line = line[-25:]
                filename = line.replace(" ", "_").replace(".", "").replace(",", "").replace("!", "") +speaker + ".wav"
                output_file_path = os.path.join(app.config['OUTPUT_NOVEL_DIR'], filename)  

                # Save the synthesized audio to a WAV file  
                with wave.open(output_file_path, "w") as fp:  
                    fp.setparams((1, 2, tts.get_sampling_rate(), len(samples), "NONE", "NONE"))  
                    fp.writeframes(samples)  

                # Load the WAV file and increase the volume  
                audio = AudioSegment.from_wav(output_file_path)  
                louder_audio = audio + 6  # Increase volume by 6 dB  
                # Save the louder audio as an MP3 file  
                mp3_filename = "louder_" + filename.replace('.wav', '.mp3')  
                louder_audio.export(os.path.join(app.config['OUTPUT_NOVEL_DIR'], mp3_filename), format="mp3")  
                #copy the results to static/ouput
                shutil.copy(os.path.join(app.config['OUTPUT_NOVEL_DIR'], mp3_filename), os.path.join('static/output', mp3_filename)) 
                #shutil.copy(os.path.join(app.config['OUTPUT_DIR'], mp3_filename), os.path.join('static/novel_audio_mp3', mp3_filename)) 
                audio_files.append(mp3_filename)  
        # Retrieve the supported speakers (you may want to keep this in the app's context)  
        supported_speakers = tts.get_speakers()  
        return render_template('novel_balacoon.html', audio_files=get_balacoon_audio_files(), supported_speakers=supported_speakers,image_files=get_balacoon_image_files())  
    # Retrieve the supported speakers (you may want to keep this in the app's context)  
    supported_speakers = tts.get_speakers()  
    return render_template('novel_balacoon.html', audio_files=get_balacoon_audio_files(), supported_speakers=supported_speakers,image_files=get_balacoon_image_files())

def get_balacoon_audio_files():  
    audio_files = glob.glob('static/novel_audio_mp3/*.mp3')
    # sort by creation date in descending order
    audio_files.sort(key=os.path.getmtime, reverse=True)
    return audio_files  

def get_balacoon_image_files():
    #image_files=glob.glob("static/novel_images/*.jpg")
    image_files=glob.glob("static/novel_images/*.jpg")+glob.glob("static/novel_images/*.png")
    # sort by creation date in descending order
    image_files.sort(key=os.path.getmtime, reverse=True)
    return image_files

@app.route('/combine_novel', methods=['POST'])
def combine_novel_audio():
    image_file = request.form.get('image_path')
    audio_file = request.form.get('audio_path')
    logit(f"Received request to combine image: {image_file} and audio: {audio_file}")

    if not image_file or not audio_file:
        logit("No image or audio file selected")
        return jsonify({'error': 'No image or audio selected'}), 400

    logit(f"Selected image: {image_file}")
    logit(f"Selected audio: {audio_file}")

    # Construct full paths for the image and audio
    #image_path = os.path.join('static/novel_images', image_file)
    #audio_path = os.path.join('static/output', audio_file)
    image_path = image_file
    audio_path = audio_file

    logit(f"Image path: {image_path}")
    logit(f"Audio path: {audio_path}")
    # Generate output filename
    output_filename = f"output_{image_file.rsplit('.', 1)[0]}_{audio_file.rsplit('.', 1)[0]}.mp4"
    output_path = os.path.join('static/novel_audio_mp3', output_filename)

    try:
        logit(f"Processing image: {image_path} and audio: {audio_path}")

        # Create video from image and audio
        output_path = add_sound_to_novel_image(image_path, audio_path)
        if output_path:
            return jsonify({'success': True, 'output_video': output_path})
        else:
            return jsonify({'error': 'Error creating video 5934'}), 500

    except Exception as e:
        logit(f"Error processing video and audio: {str(e)}")
        return jsonify({'error': str(e)}), 700

def add_sound_to_novel_image(image_path, audio_path):
    # Define the path for the silent audio clip in MP3 format
    silence_path = "silence.mp3"
    
    # Check if the silent audio file exists; if not, create a 0.7-second MP3 silence
    if not os.path.exists(silence_path):
        silence = AudioSegment.silent(duration=800)  # 700 ms of silence
        silence.export(silence_path, format="mp3")

    # Load both the main audio file and the silence clip
    audio_clip = AudioFileClip(audio_path)
    silence_clip = AudioFileClip(silence_path)

    # Apply fade-out to the audio clips before concatenation using moviepy's audio_fadeout
    fade_duration = 0.6  # Fade duration in seconds
    audio_clip = afx.audio_fadeout(audio_clip, fade_duration)
    silence_clip = afx.audio_fadeout(silence_clip, fade_duration)

    # Concatenate silence before, main audio, and silence after
    combined_audio = concatenate_audioclips([silence_clip, audio_clip, silence_clip])

    # Ensure the audio duration matches the image duration exactly
    image_clip = ImageClip(image_path).set_duration(combined_audio.duration)

    # Define the output path for the combined audio in MP3 format
    combined_audio_path = audio_path.replace('.mp3', '_combined.mp3')
    
    # Write the combined audio to a file (ensure it's saved as MP3)
    combined_audio.write_audiofile(combined_audio_path, codec='libmp3lame')

    # Set the combined audio to the image clip
    video_clip = image_clip.set_audio(combined_audio)

    # Define the output path for the final video
    output_path = image_path.replace('.jpg', '_audio.mp4').replace('.png', '_audio.mp4')

    # Write the final video file with an FPS (frames per second) of 24
    video_clip.write_videofile(output_path, codec='libx264', audio_codec='aac', fps=24)


    
    # Copy to static/temp_exp/STILLX.mp4
    shutil.copy(output_path, 'static/temp_exp/STILLX.mp4')
    im = Image.open(image_path).convert("RGB") 
    w, h = im.size
    if w > 833:im = im.resize((512, 512), resample=Image.LANCZOS)
    if w == 832:im = im.resize((512, 768), resample=Image.LANCZOS)
    im.save("static/projects/use.jpg")
    #copy to static/projects/use.mp3
    shutil.copy(audio_path, "static/projects/use.mp3")
    subprocess.run(['/home/jack/miniconda3/envs/cloned_base/bin/python','/home/jack/Desktop/Flask_Make_Art/Wav2Lip-master/makeavatar'], check=True)
    return output_path
  
@app.route('/merge_novel_videos', methods=['GET', 'POST'])  
def merge_novel_videos(): 
    mp4_files = glob.glob('static/temp_exp/*.mp4')
    mp4_files.sort(key=os.path.getmtime, reverse=True) 
    return render_template('merge_novel_videos.html', mp4_files=mp4_files)


# Directory where the videos are stored
VIDEO_DIRECTORY = 'static/temp_exp'
MERGED_VIDEO_PATH = os.path.join(VIDEO_DIRECTORY, 'merged_novel_videos.mp4')


@app.route('/merge_videos', methods=['POST'])
def merge_videos():
    # Retrieve selected video paths from the form
    selected_videos = request.form.getlist('selected_videos')
    #list last video by creation time
    selected_videos.sort(key=os.path.getmtime)#, reverse=True)
    if not selected_videos:
        return redirect(url_for('merge_novel_videos'))

    try:
        # Load each selected video using moviepy
        clips = [VideoFileClip(video) for video in selected_videos]

        # Concatenate all selected videos into one
        merged_clip = concatenate_videoclips(clips, method="compose")
        merged_clip.write_videofile(MERGED_VIDEO_PATH, codec="libx264")

        # Close all video clips
        for clip in clips:
            clip.close()

    except Exception:
        return "An error occurred while merging the videos."

    # Return the merged video file as a download
    #copy to static/temp_exp/novel.mp4
    shutil.copy(MERGED_VIDEO_PATH, 'static/temp_exp/novel.mp4')

    return send_file(MERGED_VIDEO_PATH, as_attachment=True)

# Route for the form
@app.route('/add_novel_text', methods=['GET', 'POST'])
def add_novel_text():
    # Get the list of images in the NOVEL_IMAGES
    images = os.listdir(NOVEL_DIRECTORY)
    images = [img for img in images if img.lower().endswith(('.png', '.jpg', '.jpeg'))]   
    
    if request.method == 'POST':
        image_file = request.form['image_file']
        text = request.form['text']
        position = (int(request.form['x_position']), int(request.form['y_position']))
        font_size = int(request.form['font_size'])
        color = request.form['color']
        font_path = os.path.join(app.config['FONT_FOLDER'], 'xkcd-script.ttf')
        font = ImageFont.truetype(font_path, font_size)

        # Open the image
        image_path = os.path.join('static/novel_images', image_file)
        image = Image.open(image_path)

        # Draw the text on the image
        draw = ImageDraw.Draw(image)
        draw.text(position, text, font=font, fill=color)

        # Save the temporary image for preview
        temp_image_path = os.path.join('static/temp_image', 'temp-image.png')
        image=image.convert('RGB')
        image.save(temp_image_path)
        dest=os.path.join('static/novel_images', 'temp-image.png')
        shutil.copy(temp_image_path, dest)
        return render_template('add_novel_text.html', images=images, selected_image=image_file, temp_image='temp-image.png', text=text, position=position, font_size=font_size, color=color)
    
    return render_template('add_novel_text.html', images=images)

# Route to save the final image
@app.route('/save_novel_image', methods=['POST'])
def save_novel_image():
    image_file = request.form['image_file']
    final_text = request.form['final_text']
    position = eval(request.form['final_position'])  # Convert string back to tuple
    font_size = int(request.form['final_font_size'])
    color = request.form['final_color']
    font_path = os.path.join(app.config['FONT_FOLDER'], 'xkcd-script.ttf')
    font = ImageFont.truetype(font_path, font_size)

    # Open the image again
    image_path = os.path.join('static/novel_images', image_file)
    image = Image.open(image_path)

    # Draw the final text on the image
    draw = ImageDraw.Draw(image)
    draw.text(position, final_text, font=font, fill=color)

    # Save the image with a unique UUID
    unique_filename = f"{uuid.uuid4()}.png"
    final_image_path = os.path.join(VIDEO_DIRECTORY, unique_filename)
    image.save(final_image_path)


    return redirect(url_for('add_text'))

#--------------------------------    
# Load a default font (replace with an actual font path if available)
DEFAULT_FONT_PATH = "static/fonts/DejaVuSans-Bold.ttf"
FONT_SIZE = 24
PADDING = 10
NOVEL_DIRECTORY = 'static/novel_images'
@app.route('/add_novel_caption')
def add_novel_caption():
    # Get the list of images in the NOVEL_DIRECTORY
    images = [img for img in os.listdir(NOVEL_DIRECTORY) if img.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    # Sort images by modification time (latest image last)
    images.sort(key=lambda img: os.path.getmtime(os.path.join(NOVEL_DIRECTORY, img)))
    dest=os.path.join('static/novel_images', 'captioned.jpg')
    # `captioned_image` is now the most recently modified/added image
    return render_template('add_novel_caption.html', images=images, captioned_image=dest)


@app.route('/add_caption', methods=['POST'])
def add_caption():
    # Get selected image and caption text
    selected_image = request.form.get('selected_image')
    caption_text = request.form['caption']
    # Retrieve optional offset from user input; default to 20 if not provided
    offset = int(request.form.get('offset', 20))
    alpha = int(request.form.get('alpha', 155))
    if not selected_image:
        return "No image selected.", 400
    
    # Sanitize caption text for a valid filename
    safe_caption_text = re.sub(r'[^a-zA-Z0-9_-]', '', caption_text.replace(' ', '_'))
    
    # Generate new name for the captioned image (appending '_captioned' to the original name)
    base_name, ext = os.path.splitext(selected_image)
    captioned_image_name = f"{base_name}_captioned{ext}"
    output_path = os.path.join('static/novel_images', captioned_image_name)
    dest = os.path.join('static/novel_images', 'captioned.jpg')
    
    # Open the selected image
    image_path = os.path.join(NOVEL_DIRECTORY, selected_image)
    image = Image.open(image_path).convert("RGBA")
    
    # Create an overlay for the caption
    text_image = Image.new("RGBA", image.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(text_image)
    font = ImageFont.truetype(DEFAULT_FONT_PATH, FONT_SIZE)
    
    # Split the caption text into lines by newline character
    lines = caption_text.splitlines()  # Split text into lines by newline
    
    # Calculate the width of the longest line to ensure the box is wide enough
    max_text_width = max(draw.textlength(line, font=font) for line in lines)
    line_height = FONT_SIZE + PADDING // 2  # Add space between lines
    total_text_height = line_height * len(lines)  # Total height for all lines
    
    # Set rectangular background width and height
    background_width = max_text_width + 2 * PADDING
    background_height = total_text_height + 2 * PADDING
    background_x = (image.width - background_width) // 2
    background_y = image.height - background_height - offset  # Near the bottom of the image

    # Draw white background for the caption box
    draw.rectangle(
        [background_x, background_y, background_x + background_width, background_y + background_height],
        fill=(255, 255, 255, alpha)
    )
    
    # Draw each line of text, ensuring that lines are centered and spaced out
    y_offset = background_y + PADDING  # Start drawing after padding
    for line in lines:
        # Get the width of the current line
        text_width = draw.textlength(line, font=font)
        text_x = background_x + (background_width - text_width) // 2  # Center each line
        draw.text((text_x, y_offset), line, font=font, fill=(0, 0, 0, 255))
        y_offset += line_height  # Move down for the next line
    
    # Combine captioned overlay with original image
    combined_image = Image.alpha_composite(image, text_image)
    
    # Save the final captioned image with a new name to avoid overwriting
    combined_image.convert("RGB").save(output_path, "JPEG")
    shutil.copy(output_path, dest)
    #create uuid for captioned image
    unique_filename = f"static/novel_images/{uuid.uuid4()}.jpg"
    shutil.copy(output_path, unique_filename)
    # Retrieve updated list of images
    images = [img for img in os.listdir(NOVEL_DIRECTORY) if img.lower().endswith(('.png', '.jpg', '.jpeg'))]
    return render_template('add_novel_caption.html', image_path=output_path, images=images, captioned_image=dest)

#--------------------------------
'''
@app.route('/view_files')
def view_files():
    masks = glob.glob('static/temp_exp/*.mp4')
    masks = sorted(masks, key=os.path.getmtime, reverse=True)
    filenames = [os.path.basename(mask) for mask in masks]
    mask_data = zip(masks, filenames)
    return render_template('view_files.html', videos=masks)
print("TEST LOG")
@app.route('/delete_file', methods=['POST', 'GET'])
def delete_file():
    mask_path = request.form.get('mask_path')
    print(f'XXXXXXXX {mask_path}')
    if mask_path:
        try:
            os.remove(mask_path)
            print(f"Deleted mask: {mask_path}")
        except Exception as e:
            print(f"Error deleting mask: {e}")
    return redirect(url_for('view_files'))

'''
#--------------------------------
@app.route('/view_novel_files')
def view_novel_files():
    # Get all image paths in sorted order (newest first)
    image_paths = glob.glob('static/novel_images/*.jpg')
    image_paths = sorted(image_paths, key=os.path.getmtime, reverse=True)
    return render_template('view_novel_files.html', image_paths=image_paths)

@app.route('/delete_novel_file', methods=['POST'])
def delete_novel_file():
    # Retrieve all selected image paths
    selected_image_paths = request.form.getlist('image_paths[]')
    print(f"Selected images for deletion: {selected_image_paths}")
    
    # Iterate and delete each selected image
    for image_path in selected_image_paths:
        try:
            os.remove(image_path)
            print(f"Deleted image: {image_path}")
        except Exception as e:
            print(f"Error deleting image {image_path}: {e}")
    
    # Redirect back to view the updated file list
    return redirect(url_for('view_novel_files'))

@app.route('/cp_store')
def cp_store():
    source_dir = 'static/novel_images'
    target_dir = 'static/novel_images'

    try:
        # If the target directory doesn't exist, use copytree to copy all contents
        if not os.path.exists(target_dir):
            shutil.copytree(source_dir, target_dir)
        else:
            # If the directory exists, copy each file individually
            for filename in os.listdir(source_dir):
                source_file = os.path.join(source_dir, filename)
                target_file = os.path.join(target_dir, filename)

                # Copy only files, not directories within source_dir
                if os.path.isfile(source_file):
                    shutil.copy2(source_file, target_file)

        return redirect(url_for('mk_novel'))
    except Exception as e:
        # Handle any exceptions that occur and return an error message
        return f"Error copying files: {e}"

#------------loads images from archived-images/ --------------------


@app.route('/select_images', methods=['GET', 'POST'])
def select_images():
    if request.method == 'POST':
        top_image = request.form.get('top_image')
        mask_image = request.form.get('mask_image')
        bottom_image = request.form.get('bottom_image')

        if not top_image or not mask_image or not bottom_image:
            return "Please select one top image, one mask image, and one bottom image."

        # Redirect to the blend_images route with the selected images
        return redirect(url_for('blend_images', top=top_image, mask=mask_image, bottom=bottom_image))

    image_paths = get_image_paths()
    return render_template('select_images.html', image_paths=image_paths)

@app.route('/blend_images', methods=['POST', 'GET'])
def blend_images():
    # Retrieve selected images from the form
    top_image = request.form.get('top_image')
    mask_image = request.form.get('mask_image')
    bottom_image = request.form.get('bottom_image')
    opacity = float(request.form.get('opacity', 0.5))

    # Check if all required images are provided
    if not top_image or not mask_image or not bottom_image:
        return "Please select one top image, one mask image, and one bottom image."

    # Process images
    image_paths = [top_image, mask_image, bottom_image]
    result_path = blend_images_with_grayscale_mask(image_paths, mask_image, opacity)
    mask_image ="static/masks/mask.png"
    return redirect(url_for('clean_storage_route'))#, image_paths=image_paths, mask_path=mask_image, opacity=opacity))
#render_template('blend_result_exp.html', result_image=result_path, image_paths=image_paths, mask_image=mask_image, opacity=opacity)

def blend_images_with_grayscale_mask(image_paths, mask_path, opacity):
    if len(image_paths) != 3:
        return None
    base_image_path, mask_image_path, top_image_path = image_paths
    base_image = Image.open(base_image_path)
    mask_image = Image.open(mask_path).convert('L')
    top_image = Image.open(top_image_path)
    base_image = base_image.resize((512,768), Image.LANCZOS)
    top_image = top_image.resize((512,768), Image.LANCZOS)
    mask_image = mask_image.resize((512,768), Image.LANCZOS)
    #base_image, top_image = resize_images_to_base(base_image, [base_image, top_image])[0], resize_images_to_base(base_image, [base_image, top_image])[1]
    blended_image = Image.composite(top_image, base_image, mask_image)

    unique_id = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    result_path = os.path.join(app.config['NOVEL_IMAGES'], f'result_{unique_id}.png')
    blended_image.save(result_path)


    return result_path

def get_image_paths():
    image_paths = []
    for ext in ['png', 'jpg', 'jpeg']:
        #image_paths.extend(glob.glob(os.path.join('/static/archived-images/', f'*.{ext}')))
        # image paths are static/novel_images/
        image_paths=glob.glob("static/archived-images/*.jpg")+glob.glob("static/archived-images/*.png")
    image_paths = sorted(image_paths, key=os.path.getmtime, reverse=True)
    return image_paths



@app.route('/short_out')
def short_out():
    return redirect('mk_novel')


from flask import redirect
import os
import shutil

@app.route('/move_square')
def move_square():
    # Generate a unique ID for the new resource directory
    uid = os.urandom(8).hex()
    dest = f'static/{uid}_resources'
    
    # Make sure the destination directory exists
    os.makedirs(dest, exist_ok=True)
    
    # Source directory containing files to be moved
    src = 'static/square_resources'
    
    # Move each item within src to the new destination directory
    for item in os.listdir(src):
        src_item = os.path.join(src, item)
        dest_item = os.path.join(dest, item)
        
        # Move file or directory
        if os.path.isfile(src_item):
            shutil.move(src_item, dest_item)
        elif os.path.isdir(src_item):
            shutil.move(src_item, os.path.join(dest, item))

    # Redirect after the move operation
    return redirect('copy')

# Define paths for image directories
ARCHIVE_PATH = "static/novel_images/"  # Corrected path
CROPPED_PATH = "static/cropped/"

# Ensure the cropped directory exists
os.makedirs(CROPPED_PATH, exist_ok=True)

# Route to display all images in the archive
@app.route("/crop_archive_image")
def crop_archive_image():
    # Fetch images from the archive directory
    images = glob.glob(os.path.join(ARCHIVE_PATH, "*.*"))
    return render_template("crop_archive_image.html", images=images)

# Route to handle cropping of selected image
@app.route("/crop_store_image", methods=["POST", "GET"])
def crop_store_image():
    # Get form data
    image = request.form.get("image")
    size = request.form.get("size")
    pos = request.form.get("pos")
    logit(f"Cropping_Info, image: {image}, size: {size}, pos: {pos}")
    if not image or not size or not pos:
        flash("Please provide all required fields (image, size, and position).")
        return redirect(url_for("crop_archive_image"))

    try:
        # Convert form inputs to required formats
        size = tuple(map(int, size.split(",")))
        pos = tuple(map(int, pos.split(",")))
        
        # Process and crop the image
        new_image_path = crop_store_image_function(image, size, pos)
        logit(f"Cropping_Info, new_image_path: {new_image_path}")
        flash("Image cropped and saved successfully!")
    except Exception as e:
        flash(f"Error cropping image: {e}")
        return redirect(url_for("crop_archive_image"))

    # Redirect back to the archive image list with the new image path
    return redirect(url_for("crop_archive_image", new_image=new_image_path))

# Function to crop and save the image
def crop_store_image_function(image_path, size, pos):
    # Open the image, crop, and resize it
    im = Image.open(image_path)
    cropped_image = im.crop(pos).resize(size)
    
    # Define paths for saving the new cropped image
    new_image_path = os.path.join(CROPPED_PATH, f"cropped_{os.path.basename(image_path)}")
    orig_image_path = os.path.join(ARCHIVE_PATH, f"cropped_{os.path.basename(image_path)}")
    
    # Save the cropped image
    cropped_image.save(new_image_path)
    
    # Copy and resize the saved image to archive path at 512x768
    shutil.copy(new_image_path, orig_image_path)
    subprocess.run(['mogrify', '-resize', '512x768!', orig_image_path])
    subprocess.run(['mogrify', '-resize', '512x768!', new_image_path])
    return new_image_path

#archived-images to novel_images
@app.route('/archive_to_novel')
def archive_to_novel():
    dest = 'static/novel_images'
    # Make sure the destination directory exists
    os.makedirs(dest, exist_ok=True)
    
    # Source directory containing files to be moved
    #src = 'static/archived-images'
    src = 'static/archived_resources'
    # Move each item within src to the new destination directory
    for item in os.listdir(src):
        src_item = os.path.join(src, item)
        dest_item = os.path.join(dest, item)
        
        # Move file or directory
        if os.path.isfile(src_item):
            shutil.move(src_item, dest_item)
        elif os.path.isdir(src_item):
            shutil.move(src_item, os.path.join(dest, item))

    # Redirect after the move operation
    return redirect('img_processing')
#-------------------------



def get_an_mp3():
    mp3s = random.choice(glob.glob("static/music/*.mp3"))
    return mp3s

@app.route('/blendem')
def blendem_route():
    # Define the size for resizing images
    SIZE = (512, 768)
    # Get a list of image files
    DIR = "static/novel_images/"
    image_files = sorted(glob.glob(DIR + "*.jpg")) + sorted(glob.glob(DIR + "*.png"))
    random.shuffle(image_files)
    print(f"Number of images: {len(image_files)}")

    # Create a temporary directory to store the resized images
    temp_dir = 'btemp/'
    os.makedirs(temp_dir, exist_ok=True)

    # Load and resize the images
    resized_images = []
    for image_file in image_files:
        img = cv2.imread(image_file)
        img = cv2.resize(img, SIZE)
        resized_images.append(img)

    # Create a video writer
    out_path = 'xxxxoutput.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(out_path, fourcc, 30, SIZE)

    # Keep track of video duration
    video_duration = 0

    # Create the video with fading transitions
    for i in range(len(resized_images)):
        if video_duration >= 58:  # Limit video to 58 seconds
            break

        img1 = resized_images[i]
        img2 = resized_images[(i + 1) % len(resized_images)]  # Wrap around to the first image
        step_size = 5
        for alpha in range(0, 150):  # Gradually change alpha from 0 to 100 for fade effect
            alpha /= 150.0
            blended = cv2.addWeighted(img1, 1 - alpha, img2, alpha, 0)
            out.write(blended)
            video_duration += 1 / 30  # Assuming 30 FPS

    out.release()

    # Prepare an audio clip of the same duration (58 seconds)
    audio_clip = AudioFileClip(get_an_mp3()).subclip(0, 58)

    # Load the video clip
    video_clip = VideoFileClip(out_path)

    # Set the audio of the video clip
    video_clip = video_clip.set_audio(audio_clip)

    # Load the static frame to overlay
    frame_clip = ImageClip("static/assets/blendem_frame.png", duration=video_clip.duration).set_position("center")

    # Composite the frame with the video
    final_clip = CompositeVideoClip([video_clip, frame_clip])

    # Save the final video with the overlay and music
    final_output_path = 'static/temp_exp/blendem_final_outputX.mp4'
    uid = str(uuid.uuid4())
    des = DIR.replace("/", "_")
    mp4_file = f"/home/jack/Desktop/HDD500/collections/vids/{des}{uid}.mp4"

    final_clip.write_videofile(final_output_path, codec='libx264')
    shutil.copyfile(final_output_path, mp4_file)
    return redirect(url_for('index'))
#-------------------------


@app.route('/shrink_flipbook', methods=['GET'])
def shrink_flipbook_route():
    # Directory setup
    image_directory = 'static/novel_images'
    output_directory = 'static/temp_exp'
    final_output_directory = "/home/jack/Desktop/HDD500/collections/vids/"
    os.makedirs(output_directory, exist_ok=True)
    os.makedirs(final_output_directory, exist_ok=True)

    # Get sorted image list by last modified date
    image_list = sorted(glob.glob(os.path.join(image_directory, '*.png')) + 
                        glob.glob(os.path.join(image_directory, '*.jpg')), 
                        key=os.path.getmtime)
    
    # List to store generated video paths
    video_paths = []

    for i in range(len(image_list) - 1):
        choice = randint(1, 4)
        try:
            base_image_path = image_list[i]
            image_to_paste_path = image_list[(i - 1) % len(image_list)]

            # Open base image
            base_image = Image.open(base_image_path).convert("RGBA")
            bs = base_image.size

            # Create frames for each step in transformation
            IMG_SRC = []
            for j in range(0, bs[0], 5):
                current_frame = base_image.copy()
                image_to_paste = Image.open(image_to_paste_path).convert("RGBA")
                image_to_paste = image_to_paste.resize((bs[0] - j, bs[1] - j), Image.BICUBIC)

                # Determine paste position based on choice
                x, y = (0 + j, 0 + j) if choice == 1 else (0, 0) if choice == 2 else (0 + j, 0) if choice == 3 else (0, 0 + j)
                paste_position = (x, y)

                # Check image fits within base image dimensions
                if image_to_paste.size[0] + paste_position[0] <= base_image.size[0] and \
                   image_to_paste.size[1] + paste_position[1] <= base_image.size[1]:
                    current_frame.paste(image_to_paste, paste_position, image_to_paste)
                    IMG_SRC.append(np.array(current_frame))

            # Save frames to MP4
            output_video_path = os.path.join(output_directory, f'output_video_{i}.mp4')
            imageio.mimsave(output_video_path, IMG_SRC, fps=30)
            video_paths.append(output_video_path)
        except Exception as e:
            return jsonify({"error": f"Error processing images: {e}"}), 500

    # Concatenate videos
    input_list_path = os.path.join(output_directory, "input_list.txt")
    with open(input_list_path, 'w') as input_list_file:
        for video_path in video_paths:
            input_list_file.write(f"file '{os.path.abspath(video_path)}'\n")
    
    concatenated_video_path = os.path.join(output_directory, "final_result.mp4")
    ffmpeg_command = f"ffmpeg -y -f concat -safe 0 -i {input_list_path} -c copy {concatenated_video_path}"
    subprocess.run(ffmpeg_command, shell=True, check=True)

    # Unique filename for the final video
    final_output_video_path = os.path.join(final_output_directory, str(uuid.uuid4()) + ".mp4")
    shutil.copyfile(concatenated_video_path, final_output_video_path)

    # Add title image and music
    add_shrink_title_image(final_output_video_path)

    return redirect(url_for('index'))

def add_shrink_title_image(video_path):
    hex_color = random.choice(["#A52A2A", "#ad1f1f", "#16765c", "#7a4111", "#9b1050", "#8e215d", "#2656ca"])
    video_clip = VideoFileClip(video_path)
    width, height = video_clip.size
    video_duration = video_clip.duration
    title_image_path = "/mnt/HDD500/collections/assets/flipbook_title.png"
    padded_size = (width + 50, height + 50)
    x_position = (padded_size[0] - width) / 2
    y_position = (padded_size[1] - height) / 2

    # Background color setup and composite overlay
    rgb_tuple = tuple(int(hex_color[i:i + 2], 16) for i in (1, 3, 5))
    blue_background = ColorClip(padded_size, color=rgb_tuple)
    padded_video_clip = CompositeVideoClip([blue_background, video_clip.set_position((x_position, y_position))])
    title_image = ImageClip(title_image_path).set_duration(video_clip.duration).set_position((0, -5)).resize(padded_video_clip.size)
    composite_clip = CompositeVideoClip([padded_video_clip, title_image]).set_duration(video_clip.duration)

    # Add random music
    mp3_files = glob.glob("/mnt/HDD500/collections/music_long/*.mp3")
    random.shuffle(mp3_files)
    music_clip = AudioFileClip(random.choice(mp3_files)).subclip(20, 20 + video_clip.duration).set_duration(video_clip.duration)
    composite_clip = composite_clip.set_audio(music_clip)

    output_path = 'static/temp_exp/shrink_flipbookX.mp4'
    composite_clip.write_videofile(output_path)
    unique_output_path = f"/mnt/HDD500/collections/vids/AI_Flipbook_{uuid.uuid4().hex}.mp4"
    shutil.copyfile(output_path, unique_output_path)
#-------------------------

# Directories setup
image_directory = 'static/novel_images'
output_directory = 'static/temp_exp'
final_output_directory = "/home/jack/Desktop/HDD500/collections/vids/"
os.makedirs(image_directory, exist_ok=True)
os.makedirs(output_directory, exist_ok=True)
os.makedirs(final_output_directory, exist_ok=True)



# Function to prepare the home directory
def prep_homedirectory():
    image_directory = 'static/temp_images_exp'
    logit(f"Image directory: {image_directory}")

    # Create or clear the image directory
    if os.path.exists(image_directory):
        shutil.rmtree(image_directory)
        logit(f"Cleared contents of image directory: {image_directory}")
    os.makedirs(image_directory, exist_ok=True)
    logit(f"Created image directory: {image_directory}")

    # Copy all jpg and png files from source to image_directory
    for f in os.listdir('static/novel_images'):
        if f.endswith(('.jpg', '.jpeg', '.png')):
            logit(f"Copying {f} to {image_directory}")
            shutil.copy(os.path.join('static', 'novel_images', f), image_directory)

    # Directory containing images
    image_directory = 'static/temp_images_exp'

    # Get and sort the list of image files by last modified date
    image_files = [
    os.path.join(image_directory, f) for f in os.listdir(image_directory)
    if f.endswith(('.jpg', '.jpeg', '.png'))
    ]

    # Sort by last modified time
    image_files = sorted(image_files, key=os.path.getmtime)

    # To return only filenames without full paths, use:
    image_files = [os.path.basename(f) for f in image_files]


    return image_files

# Function to create zoom effect video
def image_dir_to_zoom():
    selected_directory = 'static/temp_images_exp'
    os.makedirs(selected_directory, exist_ok=True)
    
    image_files = glob.glob(f'{selected_directory}/*.[jp][pn]g')
    if not image_files:
        logit("No images found in the directory.")
        return None

    output_video = 'generated_video_exp.mp4'
    frame_rate = 60
    zoom_increment = 0.0005
    zoom_duration = 300
    width, height = 512, 768

    ffmpeg_cmd = (
        f"ffmpeg -hide_banner -pattern_type glob -framerate {frame_rate} "
        f"-i '{selected_directory}/*.jpg' "
        f"-vf \"scale=8000:-1,zoompan=z='min(zoom+{zoom_increment},1.5)':x='iw/2':y='ih/2-4000':d={zoom_duration}:s={width}x{height},crop={width}:{height}:0:256\" "
        f"-c:v libx264 -pix_fmt yuv420p -r {frame_rate} -s {width}x{height} -y {output_video}"
    )

    logit(f"FFmpeg command: {ffmpeg_cmd}")
    try:
        subprocess.run(ffmpeg_cmd, shell=True, check=True)
        logit("Video generated successfully.")
    except subprocess.CalledProcessError as e:
        logit(f"FFmpeg command failed: {e}")
        return None

    video_name = str(uuid.uuid4()) + '_zoom_exp.mp4'
    if not os.path.exists('static/assets_exp'):
        os.makedirs('static/assets_exp')    
    shutil.copy(output_video, os.path.join('static/assets_exp', video_name))

    output_vid = os.path.join('static/assets_exp', video_name)
    logit(f"Generated video: {output_vid}")

    video_path = output_vid
    

    # Speed up the video
    #output_path = 'static/temp_exp/final_output_exp.mp4'
    cmd = [
        'ffmpeg', '-i', video_path, '-vf', 'setpts=0.3*PTS', '-c:a', 'copy', '-y', 'static/temp_exp/VideoFaster_exp.mp4'
        ]
    logit(f"Running FFmpeg command to speed up video: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    video_path = 'static/temp_exp/VideoFaster_exp.mp4'
    #add_zoom_title_image(video_path, hex_color="#A52A2A")
    #logit("Video sped up successfully.")
    #output_path = 'static/temp_exp/final_output_expX.mp4'
    return video_path


# Function to add title image and background music
def add_zoom_title_image(video_path, hex_color="#A52A2A"):
    directory_path = "static/temp_exp"
    os.makedirs(directory_path, exist_ok=True)
    
    video_clip = VideoFileClip(video_path)
    width, height = video_clip.size
    video_duration = video_clip.duration
    padded_size = (width + 80, height + 80)
    x_position = (padded_size[0] - width) / 2
    y_position = (padded_size[1] - height) / 2

    # Convert hex to RGB and create background clip
    rgb_tuple = tuple(int(hex_color[i:i+2], 16) for i in (1, 3, 5))
    background_clip = ColorClip(padded_size, color=rgb_tuple)
    padded_video_clip = CompositeVideoClip([background_clip, video_clip.set_position((x_position, y_position))]).set_duration(video_clip.duration)

    # Load and resize title image
    title_image = ImageClip("static/assets/Zoom_Images.png").resize(padded_video_clip.size).set_position((0, -5)).set_duration(video_clip.duration)
    composite_clip = CompositeVideoClip([padded_video_clip, title_image]).set_duration(video_clip.duration)

    # Load background music
    mp3_files = glob.glob("/mnt/HDD500/collections/music_long/*.mp3")
    random.shuffle(mp3_files)
    music_clip = AudioFileClip(random.choice(mp3_files)).audio_fadein(0.5).audio_fadeout(0.5).set_duration(video_clip.duration)
    composite_clip = composite_clip.set_audio(music_clip)

    output_path = 'static/temp_exp/final_output_expX.mp4'
    composite_clip.write_videofile(output_path)
    
    # Save output and speed up the video
    unique_name = f"AI_Creates_Composite_{uuid.uuid4()}.mp4"
    shutil.copyfile(output_path, f"/home/jack/Desktop/HDD500/collections/vids/{unique_name}")

    return output_path

# Combined route to run all processes in sequence
@app.route('/zoom_each', methods=['GET'])
def zoom_each_route():
    # Step 1: Prepare home directory and get shuffled images
    images = prep_homedirectory()
    if not images:
        return jsonify(error="No images available after preparation"), 500
    
    # Step 2: Generate zoom effect video
    video_path = image_dir_to_zoom()
    if not video_path:
        return jsonify(error="Zoom effect video generation failed"), 500
    
    # Step 3: Add title image and background music to the video
    final_video_path = add_zoom_title_image(video_path)
    if not final_video_path:
        return jsonify(error="Final video generation failed"), 500

    return jsonify(message="All processing completed successfully", final_video_path=final_video_path)
#----------------------------------------

def init_db():
    """Initialize the SQLite database if it doesn't exist."""
    with sqlite3.connect(DATABASE) as conn:
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS post (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT UNIQUE,
                content TEXT UNIQUE NOT NULL,
                video_filename TEXT NULL,
                image BLOB
            )
        ''')
        conn.commit()

def load_txt_files(directory):
    """Load .txt files from the directory into the SQLite database."""
    init_db() 
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    try:
        for filename in os.listdir(directory):
            if filename.endswith('.txt'):
                filepath = os.path.join(directory, filename)
                with open(filepath, 'r', encoding='utf-8') as file:
                    title = os.path.splitext(filename)[0]
                    new_content = file.read()

                    # Step 1: Check if post with this title already exists
                    cursor.execute('SELECT content FROM post WHERE title = ?', (title,))
                    existing_post = cursor.fetchone()

                    if existing_post:
                        existing_content = existing_post[0]

                        # Step 2: Compare contents
                        if new_content != existing_content:
                            # Content has changed; delete old entry and insert new one
                            cursor.execute('DELETE FROM post WHERE title = ?', (title,))
                            cursor.execute('INSERT INTO post (title, content) VALUES (?, ?)', (title, new_content))
                            print(f'Updated content for title: {title}')
                        else:
                            print(".", end="")
                    else:
                        # No existing post with this title; insert new entry
                        cursor.execute('INSERT INTO post (title, content) VALUES (?, ?)', (title, new_content))
                        print(f'Added new post: {title}')

                    conn.commit()  # Commit after each insert/update
    except sqlite3.Error as e:
        print(f'SQLite error: {e}')
    finally:
        conn.close()
#upload an img to static/novel_images
@app.route('/upload_img', methods=['POST'])
def upload_img():
    if 'file' not in request.files:

        return jsonify(error="No file found"), 400

    file = request.files['file']

    if file.filename == '':

        return jsonify(error="No file selected"), 400

    if file and allowed_file(file.filename):

        filename = secure_filename(file.filename)

        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

        return jsonify(message="File uploaded successfully"), 200

    else:

        return jsonify(error="Invalid file type"), 400

#----------------------------------------

# Initialize Balacoon TTS with the predetermined model and voice
tts = TTS("static/balacoon/en_us_hifi_jets_cpu.addon")
OUTPUT_DIR = 'static/text2audio5'

# Ensure the output directory exists
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# Select the predetermined speaker (for simplicity, the first speaker is used here)
PREDETERMINED_SPEAKER = tts.get_speakers()[3]

# Route to process a large text file
@app.route('/convert_large_text', methods=['POST', 'GET'])
def convert_large_text():
    if request.method == 'GET':
        # Render a simple form to upload the file
        return '''
        <!DOCTYPE html>
        <html>
        <head>
            <title>Upload Text File</title>
        </head>
        <body>
            <h1>Upload Text File</h1>
            <a href="/list_audio_files">list existing files</a>
            <form action="/convert_large_text" method="post" enctype="multipart/form-data">
                <input type="file" name="text_file" accept=".txt">
                <button type="submit">Convert</button>
            </form>
        </body>
        </html>
        '''
    elif request.method == 'POST':
        if 'text_file' not in request.files:
            return "No file part", 400

        text_file = request.files['text_file']
        if text_file.filename == '':
            return "No selected file", 400

        # Read and process the text file
        text_content = text_file.read().decode('utf-8')
        paragraphs = text_content.split("\n\n")  # Split on double newlines

        audio_files = []
        for idx, paragraph in enumerate(paragraphs):
            paragraph = paragraph.strip()
            if not paragraph:
                continue  # Skip empty paragraphs

            # Generate TTS samples
            samples = tts.synthesize(paragraph, PREDETERMINED_SPEAKER)
            sampling_rate = tts.get_sampling_rate()

            # Create unique filenames for the audio
            wav_filename = f"paragraph_{idx+1}.wav"
            wav_file_path = os.path.join(OUTPUT_DIR, wav_filename)

            # Save the audio as a WAV file
            with wave.open(wav_file_path, "w") as wav_file:
                wav_file.setparams((1, 2, sampling_rate, len(samples), "NONE", "NONE"))
                wav_file.writeframes(samples)

            # Convert to MP3 with volume boost
            audio = AudioSegment.from_wav(wav_file_path)
            louder_audio = audio + 6  # Increase volume by 6 dB
            mp3_filename = f"paragraph_{idx+1}.mp3"
            mp3_file_path = os.path.join(OUTPUT_DIR, mp3_filename)
            louder_audio.export(mp3_file_path, format="mp3")

            audio_files.append(mp3_filename)

        return render_template('audio_files.html', audio_files=audio_files)


# Route to list available audio files
@app.route('/list_audio_files')
def list_audio_files():
    audio_files = [f for f in os.listdir(OUTPUT_DIR) if f.endswith('.mp3')]
    audio_files.sort()
    return render_template('audio_files.html', audio_files=audio_files)

# Route to download a specific audio file
@app.route('/download_audio/<filename>')
def download_audio(filename):
    # Get list of audio and image files
    audio_files = [f for f in os.listdir('static/text2audio') if f.endswith('.mp3')]
    image_files = [f for f in os.listdir('static/novel_images') if f.endswith(('.png', '.jpg'))]
    random.shuffle(image_files)
    image_files = image_files[:50]  # Limit to 50 images
    return render_template('sound_to_story.html', audio_files=audio_files, image_files=image_files)



@app.route('/create_story', methods=['POST'])
def create_story_audio():
    image_file = request.form.get('image_path')
    audio_file = request.form.get('audio_path')
    logit(f"Received request to create story image: {image_file} and audio: {audio_file}")

    if not image_file or not audio_file:
        logit("No image or audio file selected")
        return jsonify({'error': 'No image or audio selected'}), 400

    logit(f"Selected image: {image_file}")
    logit(f"Selected audio: {audio_file}")

    # Construct full paths for the image and audio
    image_path = os.path.join('static/novel_images', image_file)
    audio_path = os.path.join('static/text2audio', audio_file)
    #image_path = image_file
    #audio_path = audio_file

    logit(f"Image path: {image_path}")
    logit(f"Audio path: {audio_path}")
    # Generate output filename
    output_filename = f"output_{image_file.rsplit('.', 1)[0]}_{audio_file.rsplit('.', 1)[0]}.mp4"
    output_path = os.path.join('static/novel_audio_mp3', output_filename)

    try:
        logit(f"Processing image: {image_path} and audio: {audio_path}")

        # Create video from image and audio
        output_path = add_sound_to_novel_image(image_path, audio_path)
        if output_path:
            return jsonify({'success': True, 'output_video': output_path})
        else:
            return jsonify({'error': 'Error creating video 5934'}), 500

    except Exception as e:
        logit(f"Error processing video and audio: {str(e)}")
        return jsonify({'error': str(e)}), 700

def add_sound_to_story(image_path, audio_path):
    logit(f"Adding sound to story: {image_path} and {audio_path}")
    # Define the path for the silent audio clip in MP3 format
    silence_path = "silence.mp3"
    
    # Check if the silent audio file exists; if not, create a 0.7-second MP3 silence
    if not os.path.exists(silence_path):
        silence = AudioSegment.silent(duration=800)  # 700 ms of silence
        silence.export(silence_path, format="mp3")

    # Load both the main audio file and the silence clip
    audio_clip = AudioFileClip(audio_path)
    silence_clip = AudioFileClip(silence_path)

    # Apply fade-out to the audio clips before concatenation using moviepy's audio_fadeout
    fade_duration = 0.6  # Fade duration in seconds
    audio_clip = afx.audio_fadeout(audio_clip, fade_duration)
    silence_clip = afx.audio_fadeout(silence_clip, fade_duration)

    # Concatenate silence before, main audio, and silence after
    combined_audio = concatenate_audioclips([silence_clip, audio_clip, silence_clip])

    # Ensure the audio duration matches the image duration exactly
    image_clip = ImageClip(image_path).set_duration(combined_audio.duration)

    # Define the output path for the combined audio in MP3 format
    combined_audio_path = audio_path.replace('.mp3', '_combined.mp3')
    
    # Write the combined audio to a file (ensure it's saved as MP3)
    combined_audio.write_audiofile(combined_audio_path, codec='libmp3lame')

    # Set the combined audio to the image clip
    video_clip = image_clip.set_audio(combined_audio)

    # Define the output path for the final video
    output_path = image_path.replace('.jpg', '_audio.mp4').replace('.png', '_audio.mp4')

    # Write the final video file with an FPS (frames per second) of 24
    video_clip.write_videofile(output_path, codec='libx264', audio_codec='aac', fps=24)


    
    # Copy to static/temp_exp/STILLX.mp4
    shutil.copy(output_path, 'static/temp_exp/STILLX.mp4')
    im = Image.open(image_path).convert("RGB") 
    w, h = im.size
    if w > 833:im = im.resize((512, 512), resample=Image.LANCZOS)
    if w == 832:im = im.resize((512, 768), resample=Image.LANCZOS)
    im.save("static/projects/use.jpg")
    #copy to static/projects/use.mp3
    shutil.copy(audio_path, "static/projects/use.mp3")
    subprocess.run(['/home/jack/miniconda3/envs/cloned_base/bin/python','/home/jack/Desktop/Flask_Make_Art/Wav2Lip-master/makeavatar'], check=True)
    return output_path

@app.route('/combine_story', methods=['POST'])
def combine_story_audio():
    image_file = request.form.get('image_path')
    audio_file = request.form.get('audio_path')

    if not image_file or not audio_file:
        logit("No image or audio file selected")
        return jsonify({'error': 'No image or audio selected'}), 400

    logit(f"Selected image: {image_file}")
    logit(f"Selected audio: {audio_file}")

    # Construct full paths for the image and audio
    image_path = os.path.join('static/novel_images', image_file)
    audio_path = os.path.join('static/text2audio', audio_file)

    # Generate output filename
    output_filename = f"output_{image_file.rsplit('.', 1)[0]}_{audio_file.rsplit('.', 1)[0]}.mp4"
    output_path = os.path.join('static/temp_exp', output_filename)

    try:
        logit(f"Processing image: {image_path} and audio: {audio_path}")

        # Create video from image and audio
        output_path = add_sound_to_story(image_path, audio_path)
        if output_path:
            return jsonify({'success': True, 'output_video': output_path})
        else:
            return jsonify({'error': ' 3961'}), 500

    except Exception as e:
        logit(f"Error processing video and audio: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/mp3upload', methods=['POST', 'GET'])
def mp3upload():
    print("Received MP3 upload request.")
    audio_file = None  # Default to no audio file

    if request.method == 'POST':
        # Check if a file part is in the request
        if 'file' not in request.files:
            print("No file part in the request.")
            return render_template('player.html', error="No file uploaded", audio_file=None)

        file = request.files['file']

        # Check if a file was selected
        if file.filename == '':
            print("No file selected for upload.")
            return render_template('playmp3.html', error="No file selected", audio_file=None)

        # Save the MP3 file
        if file and file.filename.lower().endswith('.mp3'):
            audio_file = os.path.join('static', 'TEMP.mp3')
            file.save(audio_file)
            print(f"MP3 file saved to {audio_file}")
            return render_template('playmp3.html', audio_file='TEMP.mp3')
        
        print("Invalid file type. Only MP3 files are allowed.")
        return render_template('player.html', error="Invalid file type. Please upload an MP3 file.", audio_file=None)

    return render_template('playmp3.html', audio_file=None)
#--------------------

def sound2image(image_path, audio_path):
    try:
        # Log the paths for debugging
        print(f"Image Path: {image_path}")
        print(f"Audio Path: {audio_path}")
        
        # Create an ImageClip from the image
        image_clip = ImageClip(image_path)
        
        # Load the audio file
        audio_clip = AudioFileClip(audio_path)

        # Log the duration of the audio for debugging
        print(f"Audio Duration: {audio_clip.duration} seconds")
        
        # Set the duration of the image clip to match the audio clip duration
        image_clip = image_clip.set_duration(audio_clip.duration)
        
        # Set the audio to the image clip
        video_clip = image_clip.set_audio(audio_clip)
        
        # Define the output path for the video
        output_path = image_path.replace('.jpg', '_audio.mp4').replace('.png', '_audio.mp4')
        
        # Log the output path for debugging
        print(f"Output Video Path: {output_path}")
        
        # Write the final video file with an FPS (frames per second) of 24
        video_clip.write_videofile(output_path, codec='libx264', audio_codec='aac', fps=24)
        #copy to static/temp_exp/STILLX.mp4
        shutil.copy(output_path, 'static/temp_exp/STILLX.mp4')

        'temp_exp/sound2_imageX.mp4'
            
        im = Image.open(image_path).convert("RGB") 
        w, h = im.size
        if w > 833:im = im.resize((512, 512), resample=Image.LANCZOS)
        if w == 832:im = im.resize((512, 768), resample=Image.LANCZOS)
        im.save("static/projects/use.jpg")
        #copy to static/projects/use.mp3
        shutil.copy(audio_path, "static/projects/use.mp3")
        subprocess.run(['/home/jack/miniconda3/envs/cloned_base/bin/python','/home/jack/Desktop/Flask_Make_Art/Wav2Lip-master/makeavatar'], check=True)
        
        return output_path
       

    except Exception as e:
        print(f"An error occurred: {e}")
        return None

#if 'static/story_resources' not exist
if not os.path.exists('static/story_resources'):
    os.makedirs('static/story_resources')
@app.route('/sound_2_image', methods=['POST'])
def sound_2_image():
    image_file = request.form.get('image_path')
    audio_file = request.form.get('audio_path')

    if not image_file or not audio_file:
        logit("No image or audio file selected")
        return jsonify({'error': 'No image or audio selected'}), 400

    logit(f"Selected image: {image_file}")
    logit(f"Selected audio: {audio_file}")

    # Construct full paths for the image and audio
    image_path = os.path.join('static/story_resources', image_file)
    audio_path = os.path.join('static/text2audio3', audio_file)

    # Generate output filename
    output_filename = f"output_{image_file.rsplit('.', 1)[0]}_{audio_file.rsplit('.', 1)[0]}.mp4"
    output_path = os.path.join('static/temp_exp', output_filename)

    try:
        logit(f"Processing image: {image_path} and audio: {audio_path}")

        # Create video from image and audio
        output_path = sound2image(image_path, audio_path)
        if output_path:
            return jsonify({'success': True, 'output_video': output_path})
        else:
            return jsonify({'error': ' 3961'}), 500

    except Exception as e:
        logit(f"Error processing video and audio: {str(e)}")
        return jsonify({'error': str(e)}), 500



@app.route('/sound2_image', methods=['GET'])
def sound2_image():
    # Get list of audio and image files
    audio_files = [f for f in os.listdir('static/text2audio3/') if f.endswith('.mp3')]
    #sort audio files
    # Sort audio files numerically based on the numbers in their filenames
    audio_files.sort(key=lambda x: int(os.path.splitext(x)[0]))
    image_files = [f for f in os.listdir('static/story_resources') if f.endswith(('.png', '.jpg'))]
    random.shuffle(image_files)
    image_files = image_files[:50]  # Limit to 50 images
    return render_template('mp3_2_image.html', audio_files=audio_files, image_files=image_files)

"static/story_resources/*.mp4"
@app.route('/story', methods=['GET'])
def story():
    # Get list of audio and image files

    video_files = glob.glob('static/story_resources/*.mp4')
    return render_template('story.html', video_files=video_files)
#--------------------
@app.route('/get_trends', methods=['GET', 'POST'])
def get_trends():
    pytrends = TrendReq()
    region = 'united_states'  # Default region
    keyword = 'technology'  # Default keyword
    trends = []
    suggestions = []

    if request.method == 'POST':
        # Get region and keyword from the form
        region = request.form.get('region', 'united_states')
        keyword = request.form.get('keyword', 'technology')

        # Fetch trending searches for the selected region
        trending_searches = pytrends.trending_searches(pn=region)
        trends = trending_searches[0].tolist()

        # Fetch suggestions for the keyword
        suggestions = pytrends.suggestions(keyword=keyword)

    return render_template('trends.html', trends=trends, suggestions=suggestions, region=region, keyword=keyword)

'''
@app.route("/search_templates", methods=["GET"])
def search_templates():
    # cat all templates contents into static/All.html
    subprocess.run(["cat", "templates/*.html",">","static/All.html"], cwd="templates", check=True)
    # Path to the concatenated HTML file
    html_file_path = "static/All.html"

    # Read and load the HTML file as text
    try:
        with open(html_file_path, "r", encoding="utf-8") as file:
            html_content = file.read()
    except FileNotFoundError:
        html_content = "HTML file not found."

    # Render the template and pass the HTML content
    return render_template("All.html", html_content=html_content)
'''
@app.route("/search_all_templates", methods=["GET"])
def search_all_templates():
    # Path to the static file
    txt_file_path = "static/All.txt"

    # Get all HTML files in the templates directory
    templates_dir = "templates"
    html_files = [f for f in os.listdir(templates_dir) if f.endswith(".html")]

    try:
        # Concatenate all files into All.txt with HTML content escaped
        with open(txt_file_path, "w", encoding="utf-8") as output_file:
            for html_file in html_files:
                with open(os.path.join(templates_dir, html_file), "r", encoding="utf-8") as file:
                    content = file.read()  # Read file content
                    #escaped_content = html.escape(content)  # Escape the content
                    #output_file.write(escaped_content)  # Write the escaped content to the output file
                    output_file.write(content)
                    output_file.write("\n---------------\n")  # Add a newline between files

        # Read the concatenated content from All.txt
        with open(txt_file_path, "r", encoding="utf-8") as file:
            txt_content = file.read()  # Read the escaped content

    except FileNotFoundError:
        txt_content = "Text file not found."

    # Render the template and pass the escaped content
    return render_template("template_search.html", txt_content=txt_content)
#--------------------
TITLE_DIR = 'static/overlay_zooms/title/'
MAIN_DIR = 'static/temp_exp/'
OUTPUT_PATH = 'static/temp_exp/titledX.mp4'
RESIZED_TITLE_PATH = 'static/temp_exp/resized_title.mp4'
RESIZED_MAIN_PATH = 'static/temp_exp/resized_main.mp4'


@app.route('/title', methods=['GET', 'POST'])
def title_route():
    if request.method == 'POST':
        title_file = request.form.get('title_file')
        main_file = request.form.get('main_file')

        if title_file and main_file:
            title_path = os.path.join(TITLE_DIR, title_file)
            main_path = os.path.join(MAIN_DIR, main_file)
            
            # Resize videos
            resize_videos(title_path, main_path)
            
            # Concatenate videos
            concatenate_title_video()
            video= 'static/temp_exp/titledX.mp4'
            return render_template('title.html', video=video)

    title_files = [f for f in os.listdir(TITLE_DIR) if f.endswith('.mp4')]
    main_files = [f for f in os.listdir(MAIN_DIR) if f.endswith('.mp4')]

    return render_template('title.html', title_files=title_files, main_files=main_files)


def resize_videos(title_path, main_path):
    width = 512
    height = 768

    command_title = [
        'ffmpeg', '-i', title_path,
        '-vf', f'scale={width}:{height}',
        '-c:v', 'libx264', '-preset', 'medium', '-crf', '23',
        '-y', RESIZED_TITLE_PATH
    ]

    command_main = [
        'ffmpeg', '-i', main_path,
        '-vf', f'scale={width}:{height}',
        '-c:v', 'libx264', '-preset', 'medium', '-crf', '23',
        '-y', RESIZED_MAIN_PATH
    ]

    try:
        subprocess.run(command_title, check=True)
        subprocess.run(command_main, check=True)
        print("Resizing successful!")
    except subprocess.CalledProcessError as e:
        print(f"Error during resizing: {e}")


def concatenate_title_video():
    command = [
        'ffmpeg',
        '-i', RESIZED_TITLE_PATH,
        '-i', RESIZED_MAIN_PATH,
        '-filter_complex', '[0:v:0][1:v:0]concat=n=2:v=1[outv]',
        '-map', '[outv]',
        '-c:v', 'libx264',
        '-preset', 'medium', '-crf', '23',
        '-shortest',
        '-y', OUTPUT_PATH
    ]

    try:
        subprocess.run(command, check=True)
        print("Concatenation successful!")
    except subprocess.CalledProcessError as e:
        print(f"Error during concatenation: {e}")
#--------------------

def allowed_file(filename):
    """Check if the file has an allowed extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/index_upload')
def index_upload():
    """Render the upload page."""
    return render_template('index_upload.html')

@app.route('/upload', methods=['POST', 'GET'])
def upload_file():
    """Handle the file upload."""
    if request.method == 'POST':
        # Check if the POST request has the file part
        if 'file' not in request.files:
            flash('No file part')
            print("No file part in request.")
            return redirect(request.url)

        file = request.files['file']

        # Check if a file was selected
        if file.filename == '':
            flash('No selected file')
            print("No file selected.")
            return redirect(request.url)

        # Check if the file has a valid extension
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)

            # Save the file to the main upload directory
            save_path = os.path.join('static/novel_images', filename)
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            file.save(save_path)

            # Copy the file to additional directories
            shutil.copy(save_path, 'static/image-archives/')
            shutil.copy(save_path, 'static/archived_resources/')
            print(f"File saved to {save_path} and copied to archive directories.")

            return redirect(url_for('uploaded_file', filename=filename))
        else:
            flash('File type not allowed')
            print("File type not allowed.")
            return redirect(request.url)
    return render_template('index_upload.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """Display the uploaded file."""
    file_url = url_for('static', filename=f'novel_images/{filename}')
    return f"File successfully uploaded: <a href='{file_url}'>{filename}</a>"
#--------------------
# Flask configuration for file uploads
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['ALLOWED_AUDIO_EXTENSIONS'] = {'mp3','jpg'}

def allowed_file(filename):
    """Check if the uploaded file is an allowed MP3 file."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_AUDIO_EXTENSIONS']

@app.route('/mp3_upload', methods=['POST', 'GET'])
def mp3_upload():
    if request.method == 'POST':
        # Check if the POST request has the file part
        if 'file' not in request.files:
            return 'No file part in the request', 400
        
        file = request.files['file']
        
        # If no file is selected
        if file.filename == '':
            return 'No file selected', 400
        
        # If the file is allowed
        if file and allowed_file(file.filename):
            audio_file = os.path.join(app.config['UPLOAD_FOLDER'], 'TEMP.mp3')
            file.save(audio_file)
            print(f"MP3 file saved to {audio_file}")
            # Pass the saved file path to the template
            return render_template('player.html', audio_file=audio_file)
        else:
            return 'Invalid file type. Only MP3 files are allowed.', 400
    
    # For GET requests, render the upload form
    return '''
    <!doctype html>
    <title>Upload MP3</title>
    <h1>Upload MP3 File</h1>
    <form method="POST" enctype="multipart/form-data">
        <input type="file" name="file">
        <input type="submit" value="Upload">
    </form>
    '''

@app.route('/static_audio/<filename>')
def serve_static_audio(filename):
    """Serve static files (like TEMP.mp3)."""
    return redirect(url_for('static', filename=filename))
#--------------------
@app.route('/image_upload', methods=['POST', 'GET'])
def image_upload():
    if request.method == 'POST':
        # Check if the POST request has the file part
        if 'file' not in request.files:
            return 'No file part in the request', 400
        
        file = request.files['file']
        
        # If no file is selected
        if file.filename == '':
            return 'No file selected', 400
        
        # If the file is allowed
        if file and allowed_file(file.filename):
            image_file = os.path.join(app.config['UPLOAD_FOLDER'], 'TEMP.jpg')
            file.save(image_file)
            shutil.copy(image_file,'static/story_resources/')
            print(f"MP3 file saved to {image_file}")
            # Pass the saved file path to the template
            audio_file = os.path.join(app.config['UPLOAD_FOLDER'], 'TEMP.mp3')
            shutil.copy(audio_file,'static/text2audio3')
            img = image_file
            return render_template('player.html', audio_file=audio_file, img=img)
        else:
            return 'Invalid file type. Only MP3 files are allowed.', 400
    
    # For GET requests, render the upload form
    return '''
    <!doctype html>
    <title>Upload MP3</title>
    <h1>Upload MP3 File</h1>
    <form method="POST" enctype="multipart/form-data">
        <input type="file" name="file">
        <input type="submit" value="Upload">
    </form>
    '''
#--------------------
# Flask configuration for file uploads
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['ALLOWED_AUDIO'] = {'mp3'}
app.config['ALLOWED_AUDIO_EXTENSIONS'] = {'mp3','jpg', 'jpeg', 'png'}
app.config['ALLOWED_IMAGE'] = {'jpg', 'jpeg', 'png','mp3'}

def allowed_file(filename, ALLOWED_EXTENSIONS):
    """Check if the uploaded file is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_AUDIO_EXTENSIONS']

@app.route('/combine_audio_image', methods=['POST', 'GET'])
def combine_audio_image():
    if request.method == 'POST':
        # Check if the POST request has both audio and image files
        if 'audio' not in request.files or 'image' not in request.files:
            return 'Both audio and image files are required.', 400
        
        audio_file = request.files['audio']
        image_file = request.files['image']

        # Validate file selections
        if audio_file.filename == '' or image_file.filename == '':
            return 'Both files must be selected.', 400

        if not allowed_file(audio_file.filename, app.config['ALLOWED_AUDIO']):
            return 'Invalid audio file. Only MP3 files are allowed.', 400

        if not allowed_file(image_file.filename, app.config['ALLOWED_IMAGE']):
            return 'Invalid image file. Only JPG, JPEG, or PNG files are allowed.', 400

        # Save files temporarily
        temp_audio = os.path.join(app.config['UPLOAD_FOLDER'], 'temp_audio.mp3')
        temp_image = os.path.join(app.config['UPLOAD_FOLDER'], 'temp_image.jpg')
        output_video = 'static/temp_exp/temp_videoX.mp4'

        audio_file.save(temp_audio)
        image_file.save(temp_image)
        print(f"Temporary files saved: {temp_audio}, {temp_image}")

        # Combine files into a video using FFmpeg
        ffmpeg_command = [
            'ffmpeg', '-y', '-loop', '1', '-i', temp_image,
            '-i', temp_audio, '-c:v', 'libx264', '-tune', 'stillimage',
            '-c:a', 'aac', '-b:a', '192k', '-shortest', output_video
        ]
        try:
            subprocess.run(ffmpeg_command, check=True)
            print(f"Video created successfully at {output_video}")
        except subprocess.CalledProcessError as e:
            print(f"Error creating video: {e}")
            return 'Failed to create video.', 500

        # Send the file directly to the browser for download
        output_video = 'static/temp_exp/temp_videoX.mp4'
        
        return render_template('player.html', img = temp_image ,audio_file=audio_file,video=output_video,)
    # For GET requests, render the combine form
    return '''
    <!doctype html>
    <title>Combine MP3 and JPG</title>
    <h1>Combine MP3 Audio and JPG Image into MP4</h1>
    <form method="POST" enctype="multipart/form-data">
        <label for="audio">Select MP3 file:</label><br>
        <input type="file" name="audio" id="audio"><br><br>
        <label for="image">Select JPG file:</label><br>
        <input type="file" name="image" id="image"><br><br>
        <input type="submit" value="Combine and Download">
    </form>
    '''
#--------------------


# Set up logging to capture GC debug output and store it in memory
gc_data = []  # Store GC collection data for plotting


# Enable GC debugging for memory leak detection




def continuous_gc_monitor(interval=120):
    """Run garbage collection monitoring every `interval` seconds."""
    while True:
        collected = gc.collect()  # Force garbage collection
        gc_data.append(collected)  # Store the collected data
        logit(f"Garbage collection run: {collected} objects collected")
        time.sleep(interval)  # Wait for the specified interval


@app.route('/gc_logs')
def show_gc_logs():
    """Display the last 50 lines of GC logs."""
    with open(LOG_FILE_PATH, 'r') as f:
        logs = f.readlines()
    return "<br>".join(logs[-50:])  # Show the last 50 lines of GC logs

@app.route('/gc_plot')
def gc_plot():
    """Generate and save a plot of GC collection over time as a PNG file."""
    if len(gc_data) > 1:
        # Create the plot
        fig, ax = plt.subplots()
        ax.plot(gc_data)
        ax.set(xlabel="Time Interval (seconds)", ylabel="Number of Objects Collected",
               title="Garbage Collection Over Time")
        
        # Generate a timestamped filename for the PNG
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        filename = f"gc_plot_{timestamp}.png"
        
        # Save the plot as a PNG file
        filepath = os.path.join("static", filename)
        fig.savefig(filepath)  # Save the plot as PNG in the static folder
        logit(f"Plot saved as {filepath}")

        # Return the path to the saved plot image
        return f"Garbage collection plot saved as <a href='/static/{filename}' target='_blank'>{filename}</a>"

    return "Not enough data to generate the plot."

def start_gc_monitor():
    """Start the GC monitor thread before Flask app starts serving requests."""
    thread = threading.Thread(target=continuous_gc_monitor, daemon=True)
    thread.start()
    logit("Started GC monitoring in the background.")

#--------------------
# Path to store generated images
IMAGE_FOLDER = os.path.join(os.getcwd(), 'static', 'novel_images')
if not os.path.exists(IMAGE_FOLDER):
    os.makedirs(IMAGE_FOLDER)

# Path to store generated images
IMAGE_FOLDER = os.path.join(os.getcwd(), 'static', 'novel_images')
if not os.path.exists(IMAGE_FOLDER):
    os.makedirs(IMAGE_FOLDER)

@app.route('/convert_text', methods=['GET', 'POST'])
def convert_text():
    if request.method == 'POST':
        # Get data from the form
        text = request.form['text']  # Multi-line text
        font = request.form['font']
        color = request.form['color']
        pointsize = request.form['pointsize']
        gravity = request.form['gravity']
        x = request.form['x']
        y = request.form['y']
        line_spacing = request.form['line_spacing']
        bg_color = request.form['bg_color']  # Get the selected background color
        
        # Split the text into individual lines
        text_lines = text.split('*')
        logit(f"Text lines: {text_lines}")
        # Current timestamp for file naming (using datetime.datetime)
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')  # Correct timestamp generation
        output_file = f"{IMAGE_FOLDER}/text_image_{timestamp}.jpg"

        # Start with a transparent or solid background (no white background)
        base_command = [
            'convert',
            '-size', '512x768',  # Set image size
            'xc:' + bg_color,  # Use the background color selected by the user
            output_file
        ]
        
        # Execute the base image creation (with transparent or solid color background)
        result = subprocess.run(base_command, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Error: {result.stderr}")
            raise Exception(f"ImageMagick command failed: {result.stderr}")

        # Generate the image layers for each line of text (on top of the background)
        y_base = int(y)  # Starting Y position for the first line
        for line in text_lines:
            # Add text directly to the main output file
            command = [
                'convert',
                output_file,  # Use the current image as input
                '-font', font,
                '-gravity', gravity,
                '-fill', color,
                '-pointsize', pointsize,
                '-background', bg_color,  # Set the text background color to match the image
                '-page', f"+{x}+{y_base}",  # Add the text at the specified position
                f"label:{line}",
                '-composite',  # Composite (overlay) the text on the existing image
                output_file  # Save the result back to the same file
            ]

            result = subprocess.run(command, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"Error: {result.stderr}")
                raise Exception(f"ImageMagick command failed: {result.stderr}")

            # Increment Y position for the next line
            y_base += int(pointsize) + int(line_spacing)

        # Now copy the image to a static filename so we can reference it consistently
        static_image_path = f'{IMAGE_FOLDER}/temp_image.jpg'
        shutil.copy(output_file, static_image_path)  
        # Copy timestamped file to archived-images
        shutil.copy(output_file, f'static/archived-images/text_image_{timestamp}.jpg')

        # Return the generated image with a consistent filename
        return render_template('convert_text.html', image=static_image_path)
    
    return render_template('convert_text.html')
#--------------------
@app.route('/fish_kiss')
def fish_kiss():
    return render_template('fish_kiss.html')
@app.route('/where_is_alice')
def where_is_alice():
    return render_template('where_is_alice.html')

# fish_video_maker stop_motion images at static/stop_motion
@app.route('/fish_video_maker')    
def fish_video_maker():
    return render_template('fish_video_maker.html')
 
@app.route('/stop_motion')
def stop_motion():
    return render_template('stop_motion.html')

# Ensure the save path exists
SAVE_PATH = os.path.join(app.root_path, 'static', 'capture_resources')
os.makedirs(SAVE_PATH, exist_ok=True)



@app.route('/save-capture', methods=['POST'])
def save_capture():
    try:
        # Get the image data URL from the request
        data = request.get_json()
        image_data_url = data.get('imageData')

        if not image_data_url:
            return jsonify({"error": "No image data provided"}), 400

        # Log the received data to check the image data URL
        print("Received image data:", image_data_url[:100])  # log first 100 chars

        # Remove the "data:image/png;base64," part of the URL
        image_data = image_data_url.split(",")[1]

        # Decode the base64 string to bytes
        image_bytes = base64.b64decode(image_data)

        # Create an image from the byte data
        image = Image.open(BytesIO(image_bytes))

        # Generate a unique file name
        uid = os.urandom(8).hex()
        image_path = os.path.join(SAVE_PATH, f'{uid}_captured_area.png')

        # Log the save path
        print(f"Saving image to: {image_path}")

        # Check if the directory exists
        if not os.path.exists(SAVE_PATH):
            print(f"Error: Directory does not exist: {SAVE_PATH}")
            return jsonify({"error": "Save directory does not exist"}), 500

        # Save the image
        image.save(image_path)

        shutil.copy(image_path, 'static/capture_resources/temp.png')
        # Verify the image is saved by checking the file existence
        if os.path.isfile(image_path):
            print(f"Image saved successfully at: {image_path}")
        else:
            print("Error: Image not saved to the file system.")

        # Return a success response
        #subprocess mogrify -gravity center -crop 512x512+0+0 {image_path}
        subprocess.run(['mogrify', '-gravity', 'center', '-crop', '512x512+0+0', 'static/capture_resources/temp.png'])
        #copy crop to uuid
        uid = os.urandom(8).hex()
        imagepath = os.path.join(SAVE_PATH, f'{uid}_captured_area.png')
        shutil.copy('static/capture_resources/temp.png', imagepath)
        return jsonify({"message": "Image saved successfully!", "path": image_path}), 200
    except Exception as e:
        # Log the error with full details
        print(f"Error saving image: {e}")
        return jsonify({"error": str(e)}), 500
#--------------------
#get /home/jack/Desktop/HDD500/collections/vids/ and copy to static/vids
#--------------------
# Configuration
CAPDATABASE = 'static/captions.db'
UPLOAD_CAPTIONS = 'static/captions_resources'

if not os.path.exists(UPLOAD_CAPTIONS):
    os.makedirs(UPLOAD_CAPTIONS)

# Database creation and initialization
def create_caption_db():
    conn = sqlite3.connect(CAPDATABASE)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS captions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT UNIQUE,
            width TEXT,
            height TEXT,
            text TEXT,
            background_color TEXT,
            font TEXT,
            font_size INTEGER,
            font_color TEXT
        )
    ''')
    conn.commit()
    conn.close()

# Route to serve the form
@app.route('/mk_captions')
def mk_captions():
    return render_template('mk_captions.html')

# Save caption as an image and store in database
@app.route('/save_captions', methods=['POST'])
def save_captions():
    if request.content_type != 'application/json':
        return jsonify({'success': False, 'error': 'Content-Type must be application/json'}), 415

    try:
        # Get data from the frontend (JS)
        data = request.get_json()

        # Log the data to check what we're receiving
        ic(f"Received Data: {data}")

        # Extract base64 image data
        image_data = data['imageData'].split(',')[1]  # Extract base64 image data
        image_bytes = base64.b64decode(image_data)

        # Generate a unique filename using the current timestamp
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = os.path.join(UPLOAD_CAPTIONS, f"caption_{timestamp}.png")

        # Save the image to the file system
        with open(filename, 'wb') as f:
            f.write(image_bytes)
        #SAVE COPY TO "static/novel_images" ,static/archived_resources
        shutil.copy(filename, 'static/novel_images') 
        shutil.copy(filename, 'static/archived_resources')
        shutil.copy(filename, 'static/archived-images')

        # Save data to the database
        conn = sqlite3.connect(CAPDATABASE)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO captions (
                filename, width, height, text, background_color, font, font_size, font_color
            ) 
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            filename,
            data['width'],        # Ensure box_size is passed in the form
            data['height'],
            data['text'],            # Ensure text is passed in the form
            data['background_color'], # Ensure background_color is passed in the form
            data['font'],            # Font choice passed in the form
            data['font_size'],       # Font size passed in the form
            data['font_color']       # Font color passed in the form
        ))
        conn.commit()
        conn.close()

        return jsonify({'success': True, 'filename': filename})
    except KeyError as e:
        # Handle missing fields in the form
        return jsonify({'success': False, 'error': f'Missing required field: {str(e)}'}), 400
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/get_caption_data/<int:caption_id>', methods=['GET'])
def get_caption_data(caption_id):
    """Route to retrieve caption data by ID."""
    try:
        # Connect to the database
        conn = sqlite3.connect(CAPDATABASE)
        cursor = conn.cursor()

        # Query to get the caption data by ID
        cursor.execute('SELECT * FROM captions WHERE id = ?', (caption_id,))
        caption = cursor.fetchone()

        # Close the connection
        conn.close()

        if caption:
            caption_data = {
                'id': caption[0],
                'filename': caption[1],
                'box_size': caption[2],
                'text': caption[3],
                'background_color': caption[4],
                'font': caption[6],
                'font_size': caption[7],
                'font_color': caption[8]

            }
            return jsonify({'success': True, 'caption': caption_data})
        else:
            return jsonify({'success': False, 'error': 'Caption not found'}), 404

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/view_captions')
def view_captions():
    caption_images=glob.glob('static/captions_resources/*.png')
    return render_template('view_caption_data.html',caption_images=caption_images)
#--------------------
@app.route('/transfer_directories')
def transfer_directories():
    # List directories only in 'static' folder
    transfer_src = sorted([d for d in glob.glob('static/*') if os.path.isdir(d)])
    transfer_dst = sorted([d for d in glob.glob('static/*') if os.path.isdir(d)])

    # Count files in each directory
    src_with_file_counts = [
        (d, len([f for f in os.listdir(d) if os.path.isfile(os.path.join(d, f))]))
        for d in transfer_src
    ]
    dst_with_file_counts = [
        (d, len([f for f in os.listdir(d) if os.path.isfile(os.path.join(d, f))]))
        for d in transfer_dst
    ]

    # Pass data to the template
    return render_template(
        'transfer.html',
        transfer_src=src_with_file_counts,
        transfer_dst=dst_with_file_counts,
        data_src=len(transfer_src),
        data_dst=len(transfer_dst)
    )

@app.route('/transfer_src_to_dst', methods=['POST'])
def transfer_src_to_dst():
    # Get the selected source and destination
    src = request.form.get('src')
    dest = request.form.get('dest')

    if not src or not dest:
        return "Source or destination not selected!", 400

    # Ensure the source and destination directories exist
    if not os.path.isdir(src) or not os.path.isdir(dest):
        return "Invalid source or destination directory!", 400

    # Transfer all files from source to destination
    for filename in os.listdir(src):
        src_path = os.path.join(src, filename)
        dest_path = os.path.join(dest, filename)
        if os.path.isfile(src_path):
            shutil.copy(src_path, dest_path)

    return redirect(url_for('transfer_directories'))

# Route to handle the form submission
@app.route('/empty_0000_resources')
def empty_0000_resources():
    # Get the selected source and destination
    DIR = 'static/0000_resources'
    for file in os.listdir(DIR):
        os.remove(os.path.join(DIR, file))
    return redirect(url_for('transfer_directories'))

#--------------------
# Route to show demos_and_narratives
@app.route('/demos_and_narratives')
def demos_and_narratives():
    return render_template('demos_and_narratives.html')

#--------------------
# JSON file to store video URLs
DATA_FILE = "youtube_videos.json"

# Ensure the JSON file exists
if not os.path.exists(DATA_FILE):
    with open(DATA_FILE, "w") as file:
        json.dump([], file)  # Start with an empty list


def load_youtube_videos():
    """Load video URLs from the JSON file."""
    with open(DATA_FILE, "r") as file:
        return json.load(file)


def save_youtube_videos(videos):
    """Save video URLs to the JSON file."""
    with open(DATA_FILE, "w") as file:
        json.dump(videos, file, indent=4)


# --------------------
# Route to display YouTube Videos
@app.route('/youtube_videos', methods=["GET"])
def youtube_videos():
    videos = load_youtube_videos()
    return render_template('youtube_videos.html', videos=videos)


# --------------------
# Route to save a new video URL
@app.route('/add_youtube_video', methods=["POST"])
def add_youtube_video():
    data = request.get_json()
    if not data or "video_code" not in data:
        return jsonify({"error": "Invalid data"}), 400

    video_code = data["video_code"].strip()
    if not video_code:
        return jsonify({"error": "Empty video code"}), 400

    # Load existing videos and add the new one
    videos = load_youtube_videos()
    if video_code not in videos:  # Avoid duplicates
        videos.append(video_code)
        save_youtube_videos(videos)

    return jsonify({"message": "Video added successfully", "videos": videos})


# --------------------
# Route to retrieve all videos (for API usage)
@app.route('/get_youtube_videos', methods=["GET"])
def get_youtube_videos():
    videos = load_youtube_videos()
    return jsonify(videos)

# --------------------
# Route to display YouTube Videos JSON in a textarea
@app.route('/edit_youtube_videos', methods=["GET"])
def edit_youtube_videos():
    videos = load_youtube_videos()
    return render_template('edit_youtubevideos.html', json_data=json.dumps(videos, indent=4))

# --------------------
# Route to save edited JSON data
@app.route('/save_youtube_videos', methods=["POST"])
def save_youtube_videos_api():
    data = request.form.get("json_data")
    try:
        # Validate and parse the JSON data
        videos = json.loads(data)
        if not isinstance(videos, list):
            raise ValueError("Data must be a list.")
        save_youtube_videos(videos)
        return jsonify({"message": "Videos updated successfully!"}), 200
    except (json.JSONDecodeError, ValueError) as e:
        return jsonify({"error": f"Invalid JSON: {str(e)}"}), 400
# --------------------
@app.route('/utilitie', methods=['GET', 'POST'])
def utilitie():
    if request.method == 'POST':
        filename = request.form['filename']
        filepath = 'static/text_files'
        # Ensure the filepath and filename are safe and valid
        script_path = os.path.join(filepath, filename)
        if not os.path.exists(script_path):
            flash('File not found. Please check the filename and path.', 'error')
            return redirect(url_for('utilities'))

        try:
            # Run the Python script or any command using subprocess
            if len(sys.argv) > 1:
                ARG = sys.argv[1]
                logit(f"Argument provided: {ARG}")
            else:
                ARG = ''
                logit('No argument provided')
            result = subprocess.run(['/home/jack/miniconda3/envs/cloned_base/bin/python', script_path], capture_output=True, text=True, check=True)
            output = result.stdout
            logit(f"Script {script_path} executed successfully.")
        except subprocess.CalledProcessError as e:
            output = e.stderr
            logit(f"An error occurred while executing the script {script_path}: {e}")

        return render_template('utilitie.html', result=output)

    return render_template('utilitie.html')
@app.route('/bash', methods=['GET', 'POST'])
def bash():
    if request.method == 'POST':
        filename = request.form['filename']
        arg = request.form['arg']
        filepath = 'static/text_files'
        # Ensure the filepath and filename are safe and valid
        script_path = os.path.join(filepath, filename)
        if not os.path.exists(script_path):
            flash('File not found. Please check the filename and path.', 'error')
            return redirect(url_for('bash'))

        try:
            # Run the Python script or any command using subprocess
            
            cmd=['/bin/bash', script_path,arg]
            logit(cmd)
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            output = result.stdout
            logit(f"Script {script_path} executed successfully.")
        except subprocess.CalledProcessError as e:
            output = e.stderr
            logit(f"An error occurred while executing the script {script_path}: {e}")

        return render_template('bash.html', result=output)

    return render_template('bash.html')
@app.route('/terminal_index')
def terminal_index():
    return render_template('terminal_index.html')
@app.route('/execute_command', methods=['POST'])
def execute_command():
    command = request.form.get('command')
    try:
        output = subprocess.check_output(command, shell=True, text=True, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as e:
        output = e.output
    return jsonify({'output': output})

@app.route('/index_bash')
def index_bash():
    video = findvideos()
    return render_template('run_bash.html', video=video)

def findvideo_all():
    videoroot_directory = "static"
    MP4 = []
    for dirpath, dirnames, filenames in os.walk(videoroot_directory):
        for filename in filenames:
            if filename.endswith(".mp4"): #and "Final" in filename:
                MP4.append(os.path.join(dirpath, filename))
    if MP4:
        last_video = session.get("last_video")
        new_video = random.choice([video for video in MP4 if video != last_video])
        session["last_video"] = new_video
        return new_video
    else:
        return None

@app.route('/run_bash', methods=['POST', 'GET'])
def run_bash():
    bash_command = request.form.get('bash_command')
    
    try:
        result = subprocess.check_output(bash_command, shell=True, text=True, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as e:
        result = e.output
    video = findvideos()
    return render_template('run_bash.html', result=result, video=video)

def findvideos():
    videoroot_directory = "static"
    MP4 = []
    for dirpath, dirnames, filenames in os.walk(videoroot_directory):
        for filename in filenames:
            if filename.endswith(".mp4"): #and "Final" in filename:
                MP4.append(os.path.join(dirpath, filename))
    if MP4:
        last_video = session.get("last_video")
        new_video = random.choice([video for video in MP4 if video != last_video])
        session["last_video"] = new_video
        return new_video
    else:
        return None


@app.route("/editor", methods=["GET", "POST"])
def editor():
    if request.method == "POST":
        filename = request.form["filename"]
        text = request.form["text"]
        save_text_to_file(filename, text)
        return redirect(url_for("editor"))
    else:
        #files = os.listdir(TEXT_FILES_DIR)
        #files = files.sort(key=lambda x: os.path.getmtime(x))
        files = sorted(
                [file for file in os.listdir(TEXT_FILES_DIR) if os.path.isfile(os.path.join(TEXT_FILES_DIR, file))],
                key=lambda x: os.path.getmtime(os.path.join(TEXT_FILES_DIR, x)),reverse=True)

        logit(files)
        #files = sort_files_by_date(TEXT_FILES_DIR)
        return render_template("editor1.html", files=files)

logit("you are here line 9580")
if __name__ == "__main__":
    create_caption_db()
    # Start the GC monitoring thread independently
    #start_gc_monitor()

    #start_tensorflow_server()
    # cd /mnt/HDD500/TENSORFLOW/Models && python -m http.server 8000
    #start_memory_logger(interval=60)
    #bak('app.py')
    #remove log file
    '''
    try:
        log_file = "static/app_log.txt"
        dataout = open(log_file, 'r').read()
        if len(dataout) > 100000:
            open(log_file, 'w').close()
        print("Log file cleared.")
    except: pass
    '''
    #directory = 'static/TEXT'
    #load_txt_files(directory)
    app.run(debug=True, host='0.0.0.0', port=5000)

    '''
    with app.test_request_context():
        for rule in app.url_map.iter_rules():
            methods = ','.join(rule.methods)
            logit("----------------------------------")
            logit(f"Endpoint: {rule.endpoint}, Route: {rule}, Methods: {methods}")
            logit("----------------------------------")
    '''    