# generate_pins.py
# app.py irukka same folder la itha vaikkanum
# Run: python generate_pins.py
# Output: static/pin_images/ folder la 90 hybrid images save aagum

import cv2
import numpy as np
import os

def generate_number_image(text, size=(300, 300)):
    # White background-la black text create pannum
    img = np.ones((size[1], size[0]), dtype="uint8") * 255
    font = cv2.FONT_HERSHEY_SIMPLEX
    # Text-a center-la vaikkurom
    text_size = cv2.getTextSize(str(text), font, 5, 15)[0]
    text_x = (size[0] - text_size[0]) // 2
    text_y = (size[1] + text_size[1]) // 2
    cv2.putText(img, str(text), (text_x, text_y), font, 5, (0, 0, 0), 15, cv2.LINE_AA)
    return img

def create_illusion_pin(close_num, far_num):
    img_close = generate_number_image(close_num)
    img_far   = generate_number_image(far_num)

    # Thoorathula theriya blur (Low Frequency)
    far_blur = cv2.GaussianBlur(img_far, (31, 31), 10)

    # Kitta theriya edges (High Frequency)
    close_blur  = cv2.GaussianBlur(img_close, (31, 31), 3)
    close_edges = cv2.subtract(img_close, close_blur)

    # Rendayum mix panrom
    hybrid = cv2.add(far_blur, close_edges)
    return hybrid

# Output folder
OUT_DIR = os.path.join("static", "pin_images")
os.makedirs(OUT_DIR, exist_ok=True)

count = 0
for close_num in range(10):
    for far_num in range(10):
        if close_num == far_num:
            continue  # Same digit — illusion vedam

        hybrid = create_illusion_pin(close_num, far_num)
        filename = f"{close_num}_vs_{far_num}.png"
        cv2.imwrite(os.path.join(OUT_DIR, filename), hybrid)
        count += 1
        print(f"Created: {filename}  (Close={close_num}, Far={far_num})")

print(f"\nDone! {count} images saved to '{OUT_DIR}/'")
print("Now run: python app.py")