import numpy as np
from PIL import Image

def load_image(file_path):
    img = Image.open(file_path).convert('L')  
    return np.array(img)

def save_and_show_image(image_array, file_path, title="Image"):
    img = Image.fromarray(image_array.astype(np.uint8))
    img.save(file_path)
    print(f"{title} saved as {file_path}")
    img.show()  

def dilate(image, radius=1):
    img_height, img_width = image.shape
    output = np.zeros_like(image)

    for i in range(radius, img_height - radius):
        for j in range(radius, img_width - radius):
            max_value = 0
            # Przejście po sąsiadach
            for x in range(-radius, radius + 1):
                for y in range(-radius, radius + 1):
                    if image[i + x, j + y] > max_value:
                        max_value = image[i + x, j + y]
            output[i, j] = max_value
    
    return output
