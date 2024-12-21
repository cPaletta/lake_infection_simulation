import numpy as np
from PIL import Image

def load_image(file_path):
    img = Image.open(file_path).convert('L')  # Konwersja do skali szarości
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
            for x in range(-radius, radius + 1):
                for y in range(-radius, radius + 1):
                    if image[i + x, j + y] > max_value:
                        max_value = image[i + x, j + y]
            output[i, j] = max_value
    
    return output

def erode(image, radius=1):
    img_height, img_width = image.shape
    output = np.zeros_like(image)

    for i in range(radius, img_height - radius):
        for j in range(radius, img_width - radius):
            min_value = 255
            for x in range(-radius, radius + 1):
                for y in range(-radius, radius + 1):
                    if image[i + x, j + y] < min_value:
                        min_value = image[i + x, j + y]
            output[i, j] = min_value
    
    return output

def open_morphology(image, radius=1):
    eroded = erode(image, radius)
    opened = dilate(eroded, radius)
    return opened

def close_morphology(image, radius=1):
    dilated = dilate(image, radius)
    closed = erode(dilated, radius)
    return closed

def apply_morphological_operations(image_path, output_path):
    image = load_image(image_path)
    
    # Wykonujemy operacje morfologiczne (otwarcie i zamknięcie), aby wyczyścić obszary wodne
    opened_image = open_morphology(image, radius=3)
    closed_image = close_morphology(opened_image, radius=3)
    
    save_and_show_image(closed_image, output_path, title="Processed Water Map")

# Przykładowe użycie
apply_morphological_operations("D:\Studia\modelowanie_dyskretne\lab_05\lab-5-cPaletta\mapa.png", "D:\Studia\modelowanie_dyskretne\lab_05\lab-5-cPaletta\mapa1.png")
