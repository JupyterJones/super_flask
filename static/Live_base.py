import random
import time
import numpy as np
from PIL import Image, ImageOps, ImageFilter, ImageDraw
from flask import Flask, request, render_template, redirect, url_for, send_from_directory, Response, send_file
import os
import glob
import datetime
import inspect

app = Flask(__name__, template_folder='templates_new', static_folder='static_new')

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
        with open("log.txt", "a") as file:
            file.write(log_message)

        # Print the log message to the console
        print(log_message)
    except Exception as e:
        print(f"Error occurred while logging: {e}")

def ensure_dir_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)

# Ensure necessary directories exist
ensure_dir_exists("static_new/archived-images")
ensure_dir_exists("static_new/blot")        
ensure_dir_exists("static_new/images")        

def draw_blob(img, point, size, color):
    """Draw a blob on the image."""
    draw = ImageDraw.Draw(img)
    draw.ellipse([point[0], point[1], point[0] + size, point[1] + size], fill=color)

def process_image(seed_count, seed_max_size, imgsize=(510, 766), count=0):
    """Generates a Rorschach-style inkblot and saves it."""
    margin_h, margin_v = 60, 60
    color = (0, 0, 0)
    img = Image.new("RGB", imgsize, "white")

    logit(f"Starting image generation with {seed_count} seeds and max size {seed_max_size}.")
    
    try:
        # Create the inkblot by drawing random blobs
        for seed in range(seed_count):
            point = (random.randrange(0 + margin_h, imgsize[0] // 2),
                     random.randrange(0 + margin_v, imgsize[1] - margin_v))

            blob_size = random.randint(25, seed_max_size)
            draw_blob(img, point, blob_size, color)

        # Symmetry: Flip left half onto right half
        cropped = img.crop((0, 0, imgsize[0] // 2, imgsize[1]))
        flipped = cropped.transpose(Image.FLIP_LEFT_RIGHT)
        img.paste(flipped, (imgsize[0] // 2, 0))

        # Apply Gaussian blur to the image
        blurred_img = img.filter(ImageFilter.GaussianBlur(radius=15))

        # Convert the blurred image to grayscale
        im_grey = blurred_img.convert('L')

        # Calculate the mean intensity for binarization
        mean = np.mean(np.array(im_grey))

        # Binarize the image using the mean threshold
        image_array = np.array(im_grey)
        binary_image = np.where(image_array > mean, 255, 0).astype(np.uint8)

        # Save the final normal image
        final_filename = time.strftime("static_new/archived-images/GOODblots%Y%m%d%H%M%S.png")
        ImageOps.expand(Image.fromarray(binary_image), border=1, fill='white').save(final_filename)

        # Create and save the inverted image (black and white swapped)
        inverted_image = np.where(binary_image == 255, 0, 255).astype(np.uint8)
        inverted_filename = time.strftime("static_new/archived-images/INVERTEDblots%Y%m%d%H%M%S.png")
        ImageOps.expand(Image.fromarray(inverted_image), border=1, fill='black').save(inverted_filename)

        logit(f"Images generated: {final_filename}, {inverted_filename}")
        return final_filename, inverted_filename
    except Exception as e:
        logit(f"Error during image processing: {e}")

@app.route('/rorschach')
def rorschach_route():
    """Route for generating and displaying inkblots."""
    ensure_dir_exists("static_new/images")
    ensure_dir_exists("static_new/archived-images")

    inkblot_images = []
    
    # Call the image processing function (modify parameters as needed)
    process_image(seed_count=15, seed_max_size=200)

    # Retrieve all previously created inkblots from the directory
    normal_files = glob.glob('static_new/archived-images/GOODblots*.png')
    inverted_files = glob.glob('static_new/archived-images/INVERTEDblots*.png')
    
    # Sort the file lists to keep them aligned by modification time
    # Last image first
    normal_files.sort(key=os.path.getmtime, reverse=True)
    inverted_files.sort(key=os.path.getmtime, reverse=True)

    # Pair normal and inverted images together
    for normal, inverted in zip(normal_files, inverted_files):
        inkblot_images.append({
            'normal': url_for('static', filename=normal.replace('static_new/', '')),
            'inverted': url_for('static', filename=inverted.replace('static_new/', ''))
        })

    logit(f"Retrieved {len(inkblot_images)} inkblot images for display.")
    
    # Pass the image paths to the template
    return render_template('Rorschach_1.html', inkblot_images=inkblot_images)

@app.route('/') # Home page
def home():
    return render_template('home.html')

#view for Explain.html
@app.route('/explain')
def explain():
    return render_template('Explain.html')
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
                logit(f"Matching Paragraphs:  {matching_paragraphs}")
                return render_template('notes_app.html', text=matching_paragraphs)
            else:
                return render_template('notes_app.html', text=["No matching results."])
        else:
            return render_template('notes_app.html', text=["Enter a search term."])

    return render_template('notes_app.html', text=[])

@app.route('/search', methods=['POST', 'GET'])
def search():
    if request.method == 'POST':
        search_term = request.form.get('search', '').strip()
        if search_term:
            with open('static_new/text/notes_app.txt', 'r') as f:
                text = f.read()
                paragraphs = text.split('----------')

                # Filter paragraphs that contain the search term
                matching_paragraphs = [p for p in paragraphs if search_term in p]

            if matching_paragraphs:
                logit(f"Matching Paragraphs: , {matching_paragraphs}")
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

@app.route('/', methods=['POST', 'GET'])
def index():
    if request.method == 'POST':
        search_term = request.form.get('search', '').strip()
        if search_term:
            text = read_notes()
            paragraphs = text.split('----------')

            # Filter paragraphs that contain the search term
            matching_paragraphs = [p for p in paragraphs if search_term in p]

            if matching_paragraphs:
                logit(f"Matching Paragraphs: , {matching_paragraphs}")
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
if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5100)
