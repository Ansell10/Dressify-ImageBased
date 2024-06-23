from flask import Flask, render_template, send_from_directory
from flask import request
from flask import send_file
from ultralytics import YOLO
from PIL import Image
import io
import requests
import tempfile

app = Flask(__name__)

# Load your YOLO model
model = YOLO(r"https://storage.googleapis.com/model-dressify-bucket/best.pt")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process_image():
    # Process the image here
    image_url = request.form['image_url']
    image_cloth_url = request.form['image_cloth_url']

    # Download the image
    response = requests.get(image_url)
    image = Image.open(io.BytesIO(response.content))

    response_cloth = requests.get(image_cloth_url)
    image_cloth = Image.open(io.BytesIO(response_cloth.content))

    results = model.predict(image_url)
    boxes = results[0].boxes.xyxy.tolist()
    result = results[0]
    bounding_height = int(boxes[0][3] - boxes[0][1])
    bounding_width = int(boxes[0][2] - boxes[0][0])
    masks = result.masks
    mask1 = masks[0]
    polygon = mask1.xy[0]
    mask = mask1.data[0].numpy()
    mask_img = Image.fromarray(mask, "I")
    mask_img = mask_img.convert('RGB')

    # Get image size
    width, height = image.size
    resized_mask_img = mask_img.resize((width, height), Image.LANCZOS)
    width, height = resized_mask_img.size
    resized_mask_img_pixels = resized_mask_img.load()
    white_area = (int(boxes[0][0]), int(boxes[0][1]), int(boxes[0][2]), int(boxes[0][3]))
    scale_factor = 1.05

    # Calculate the expansion size
    expand_width = int(((white_area[2] - white_area[0]) * (scale_factor - 1)) / 2)
    expand_height = int(((white_area[3] - white_area[1]) * (scale_factor - 1)) / 2)

    # Adjust the coordinates
    left = (white_area[0] - expand_width)
    top = (white_area[1] - expand_height)
    right = (white_area[2] + expand_width)
    bottom = (white_area[3] + expand_height)

    # Crop the adjusted white area
    white_part = resized_mask_img.crop(box=(white_area[0], white_area[1], white_area[2], white_area[3]))
    white_part_resized = white_part.resize((right - left, bottom - top), Image.LANCZOS)

    # Create a new blank image that will show changes
    new_img = Image.new('RGB', resized_mask_img.size, (0, 0, 0))
    new_img.paste(white_part_resized, (left, top))
    pixels = new_img.load()

    threshold_value = 128
    for y in range(height):
        for x in range(width):
            # Get pixel value (for RGB, take only one component as originally grayscale)
            r, g, b = pixels[x, y]  # All r, g, b values are the same because originally grayscale
            if r < threshold_value:
                pixels[x, y] = (0, 0, 0)  # Set pixel to black
            else:
                pixels[x, y] = (255, 255, 255)

    adjust_cloth = 150
    cloth_resized = image_cloth.resize((bounding_width + adjust_cloth, bounding_height + adjust_cloth), Image.LANCZOS)
    mask_pixels = new_img.load()
    width_mask, height_mask = new_img.size
    cloth_resized_pixels = cloth_resized.load()
    width_cloth, height_cloth = cloth_resized.size
    start_x = int(boxes[0][0]) - int(adjust_cloth / 2)
    end_x = int(boxes[0][2]) + int(adjust_cloth / 2)
    start_y = int(boxes[0][1]) - int(adjust_cloth / 2)
    end_y = int(boxes[0][3]) + int(adjust_cloth / 2)
    for y in range(start_y, end_y - 1):
        for x in range(start_x, end_x - 1):
            current_pixel = mask_pixels[x, y]
            if current_pixel == (0, 0, 0):
                mask_pixels[x, y] = mask_pixels[x, y]
            else:
                mask_pixels[x, y] = cloth_resized_pixels[x - start_x, y - start_y]

    final_masking = new_img.copy()
    final_masking_pixels = final_masking.load()

    final_image_ori = image
    final_image_ori_pixels = final_image_ori.load()

    for y in range(height):
        for x in range(width):
            current_pixel_2 = final_masking_pixels[x, y]
            if current_pixel_2 == (0, 0, 0):
                final_masking_pixels[x, y] = final_image_ori_pixels[x, y]

    # Save the final masked image to a temporary file or memory stream
    output_buffer = io.BytesIO()
    final_masking.save(output_buffer, format='JPEG')
    output_buffer.seek(0)

    return send_file(output_buffer, mimetype='image/jpeg', as_attachment=True, download_name='final_image.jpg')

if __name__ == '__main__':
    app.run(debug=True)
