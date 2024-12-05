from flask import Flask, render_template, request, redirect, url_for
import os
from PIL import Image
import tensorflow as tf
import numpy as np
import tensorflow_hub as hub
import time
import random
from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()

app = Flask(__name__)

# Directories
CONTENT_DIR = 'static/archived_resources'
STYLE_DIR = 'static/archived_resources'
RESULT_DIR = 'static/archived-store'

if not os.path.exists(CONTENT_DIR):
    os.makedirs(CONTENT_DIR)
if not os.path.exists(RESULT_DIR):
    os.makedirs(RESULT_DIR)

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

@app.route('/', methods=['GET', 'POST'])
def index():
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
        result_image.save(result_path)

        return render_template('styling.html', content_images=os.listdir(CONTENT_DIR),
                               style_images=os.listdir(STYLE_DIR),
                               result_image=result_filename)

    # List content and style images
    content_images = os.listdir(CONTENT_DIR)
    style_images = os.listdir(STYLE_DIR)

    return render_template('styling.html', content_images=content_images, style_images=style_images)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5300)
