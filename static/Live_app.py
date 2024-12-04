import random
import time
import numpy as np
from PIL import Image, ImageOps, ImageFilter
import cv2
from flask import Flask, render_template
import os

# Initialize the Flask app with the custom template directory
app = Flask(__name__, template_folder='templates_new')

def ensure_dir_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)

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

    return final_filename, inverted_filename

@app.route('/')
def rorschach():
    ensure_dir_exists("static/images")
    ensure_dir_exists("static/blot")

    # Generate the inkblots
    inkblot_images = []
    for count in range(2):  # Generate 2 inkblots as an example
        seed_count = random.randint(6, 10)
        seed_max_size = random.randint(100, 400)  # Smaller sizes to generate blobs
        final_image, inverted_image = processr_image(seed_count, seed_max_size, count=count)
        inkblot_images.append({'normal': final_image, 'inverted': inverted_image})

    # Pass the image paths to the template
    return render_template('Rorschach_1.html', inkblot_images=inkblot_images)
if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)