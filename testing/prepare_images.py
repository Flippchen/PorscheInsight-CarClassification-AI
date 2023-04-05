import io
import os
from PIL import Image, ImageOps
from rembg import remove, new_session
from PIL.Image import Image as PILImage


def replace_background(im: PILImage, post_process_mask=False, session=None) -> PILImage:
    # if not isinstance(im, PILImage):
    #   im = Image.open(io.BytesIO(im))
    session = session or new_session("u2netp")
    im = remove(im, post_process_mask=post_process_mask, session=session)

    new_im = Image.new('RGBA', im.size, 'WHITE')
    new_im.paste(im, mask=im)

    bio = io.BytesIO()
    new_im.save(bio, format='PNG')
    im_bytes = bio.getvalue()
    image = Image.open(io.BytesIO(im_bytes))
    image = image.convert('RGB')

    return image


def resize_image(image, size):
    return image.resize(size)


def resize_and_pad_image(image: PILImage, target_size: tuple):
    # Calculate the aspect ratio of the image
    aspect_ratio = float(image.width) / float(image.height)

    # Calculate the dimensions of the new image
    if aspect_ratio > 1:
        new_width = target_size[0]
        new_height = int(target_size[1] / aspect_ratio)
    else:
        new_width = int(target_size[0] * aspect_ratio)
        new_height = target_size[1]

    # Resize the image while maintaining its aspect ratio
    resized_image = image.resize((new_width, new_height), Image.ANTIALIAS)

    # Calculate padding
    padding_width = target_size[0] - new_width
    padding_height = target_size[1] - new_height

    # Calculate left and top padding to center the image
    left_padding = padding_width // 2
    top_padding = padding_height // 2

    # Pad the image to make it a square and center it
    padded_image = ImageOps.expand(resized_image, (left_padding, top_padding, padding_width - left_padding, padding_height - top_padding), fill=(0, 0, 0))

    return padded_image


def load_and_remove_bg(path, size):
    image = Image.open(path)
    # image = resize_image(image, size)
    image = resize_and_pad_image(image, size)
    image = replace_background(image)

    return image


def remove_bg_from_all_images(folder: str):
    for image in os.listdir(f'{folder}'):
        print("Removing background from", image)
        img = load_and_remove_bg(f"{folder}/{image}", (300, 300))
        img.save(f"{folder}/{image}")


if __name__ == '__main__':
    folder_path = 'test_images'
    remove_bg_from_all_images(folder_path)
