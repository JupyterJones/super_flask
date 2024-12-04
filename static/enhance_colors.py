from PIL import Image, ImageEnhance
import numpy as np
from sys import argv
import uuid

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
    
    # Step 6: Save the result image
    uid = str(uuid.uuid4())
    output_uid = f'static/archived_resources/{uid}_output_primary_colors.jpg'
    output_path = 'static/archived_resources/enhanced.jpg'
    result_image.save(output_path)
    shutil.copy(output_path, output_uid)
    print(f'Saved output to: {output_path}')

if __name__ == '__main__':
    convert_to_mellow_colors(argv[1])
