import os
from PIL import Image
from rembg import remove, new_session
import random
from joblib import Parallel, delayed, parallel_backend


def random_rgba_color():
    r = random.randint(0, 255)
    g = random.randint(0, 255)
    b = random.randint(0, 255)
    a = 255
    return (r, g, b, a)


def process_image(image_path):
    bg_image = random.choice(bg_images)

    local_image = bg_image.copy()
    image = Image.open(image_path)
    image = remove(image)

    local_image.paste(image, mask=image)
    local_image = local_image.convert("RGB")
    print(f"Saving image to {image_path}")
    local_image.save(image_path)


def get_background_images(path):
    images = []
    for root, _, files in os.walk(path):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.jfif')):
                image_path = os.path.join(root, file)
                image = Image.open(image_path)
                image = image.convert("RGB")
                image = image.resize((300, 300))
                images.append(image)

    return images


def process_images_in_folder(main_folder, n_jobs=4):
    image_paths = []
    for root, _, files in os.walk(main_folder):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                image_path = os.path.join(root, file)
                image_paths.append(image_path)

    with parallel_backend("threading", n_jobs=n_jobs):
        Parallel()(delayed(process_image)(path) for path in image_paths)


main_folder_path = r"C:\Users\phili\.keras\datasets\resized_DVM\pre_filter\porsche"
bg_folder_path = r"C:\Users\phili\.keras\datasets\bg"

bg_images = get_background_images(bg_folder_path)

process_images_in_folder(main_folder_path)
