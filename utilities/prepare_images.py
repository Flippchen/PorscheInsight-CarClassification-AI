import io
import os

import numpy as np
from scipy.ndimage import maximum_filter
from PIL import Image, ImageOps, ImageFile, ImageChops
from rembg import remove, new_session
from PIL.Image import Image as PILImage
from typing import Tuple
from functools import cache

ImageFile.LOAD_TRUNCATED_IMAGES = True


@cache
def get_session() -> new_session:
    """
    Get the U-2 Net session with caching.

    This function is decorated with @cache which means the result is stored
    and reused instead of calling the function every time.

    Returns:
        new_session: The U-2 Net session.
    """

    return new_session("u2net")


def replace_background(im: PILImage, post_process_mask=True, session=None, size: Tuple = None) -> Tuple[PILImage, PILImage]:
    """
    Replace the background of the given image with a black color.

    Args:
        im (PILImage): The input image.
        post_process_mask (bool, optional): If True, perform post-processing on the mask.
        session (optional): The U-2 Net session. If None, a new session is created.
        size (tuple, optional): The target size for the output image.

    Returns:
        Tuple[PILImage, PILImage]: The image with the replaced background and the mask.
    """

    size = size or (300, 300)
    # if not isinstance(im, PILImage):
    #   im = Image.open(io.BytesIO(im))
    session = session or get_session()
    im = remove(im, post_process_mask=post_process_mask, session=session)
    mask = im.copy()
    im = resize_cutout(im, size)

    new_im = Image.new('RGBA', im.size, 'BLACK')
    new_im.paste(im, mask=im)

    bio = io.BytesIO()
    new_im.save(bio, format='PNG')
    im_bytes = bio.getvalue()
    image = Image.open(io.BytesIO(im_bytes))
    image = image.convert('RGB')

    return image, mask


def get_bounding_box(im: PILImage) -> Tuple:
    """
    Get the bounding box of the non-transparent content in the given image.

    Args:
        im (PILImage): The input image.

    Returns:
        tuple: The bounding box (left, top, right, bottom).
    """

    # Get the data of the image
    im_data = im.getdata()

    # Get the dimensions of the image
    width, height = im.size

    # Find the bounding box
    left, top, right, bottom = width, height, 0, 0
    for y in range(height):
        for x in range(width):
            if im_data[y * width + x][3] > 0:  # If the pixel is not fully transparent
                left = min(left, x)
                top = min(top, y)
                right = max(right, x)
                bottom = max(bottom, y)

    return left, top, right, bottom


def resize_cutout(im: PILImage, size: Tuple = (300, 300)) -> PILImage:
    """
    Resize the given image to a specified size while maintaining the content's aspect ratio.

    Args:
        im (PILImage): The input image.
        size (tuple): The target size for the output image.

    Returns:
        PILImage: The resized image.
    """

    # Get the bounding box of the non-transparent content
    left, top, right, bottom = get_bounding_box(im)
    # Crop the image to the bounding box
    im_cropped = im.crop((left, top, right, bottom))

    im_resized = resize_and_pad_image(im_cropped, size, )  # fill_color=(255, 255, 255, 255)

    return im_resized


def resize_image(image, size) -> PILImage:
    """
    Resize the given image to a specified size.

    Args:
        image: The input image.
        size: The target size for the output image.

    Returns:
        image: The resized image.
    """

    return image.resize(size)


def resize_and_pad_image(image: PILImage, target_size: Tuple, fill_color=(0, 0, 0, 0)):
    """
    Resize the given image to a target size while maintaining the aspect ratio.
    Pad the image if needed to make it square and center it.

    Args:
        image (PILImage): The input image.
        target_size (tuple): The target size for the output image.
        fill_color (tuple, optional): The color to use for padding.

    Returns:
        PILImage: The resized and padded image.
    """

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
    resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)

    # Calculate padding
    padding_width = target_size[0] - new_width
    padding_height = target_size[1] - new_height

    # Calculate left and top padding to center the image
    left_padding = padding_width // 2
    top_padding = padding_height // 2

    # Pad the image to make it a square and center it
    padded_image = ImageOps.expand(resized_image, (left_padding, top_padding, padding_width - left_padding, padding_height - top_padding), fill=fill_color)

    return padded_image


def fix_image(image) -> PILImage:
    """
    Fix the orientation of the given image and convert it to RGB if not already.

    Args:
        image: The input image.

    Returns:
        image: The fixed image.
    """

    # Convert image to RGB if not already
    if image.mode != "RGB":
        image = image.convert("RGB")
    # Fix orientation if necessary
    image = ImageOps.exif_transpose(image)
    return image


def convert_mask(mask, color=(29, 132, 181), border_color=(219, 84, 97), border_fraction=0.03):
    """
    Convert the given mask to a specific color and add a border.

    Args:
        mask: The input mask.
        color (tuple, optional): The color to use for the mask.
        border_color (tuple, optional): The color to use for the border.
        border_fraction (float, optional): The fraction of the smaller dimension to use for the border size.

    Returns:
        mask_with_border: The mask with the converted color and added border.
    """

    # Convert the image to RGBA if it is not already
    if mask.mode != 'RGBA':
        mask = mask.convert('RGBA')

    border_size = int(min(mask.size) * border_fraction)

    if border_size % 2 == 0:
        border_size += 1

    # Create a copy of the mask and expand it to create the border
    mask_np = np.array(mask)
    border_mask_np = maximum_filter(mask_np, size=border_size)

    # Convert back to PIL image
    border_mask = Image.fromarray(border_mask_np)

    # Create the border by subtracting the original mask from the expanded mask
    border = ImageChops.difference(border_mask, mask)

    # Convert the mask and the border to the desired colors
    mask_np = np.array(mask)
    border_np = np.array(border)

    # Mask the areas where alpha channel is not zero
    mask_area = mask_np[..., 3] != 0
    border_area = border_np[..., 3] != 0

    # Replace RGB channels with desired color while keeping alpha channel the same
    mask_np[mask_area, :3] = color
    mask_np[mask_area, 3] = mask_np[mask_area, 3] // 4
    border_np[border_area, :3] = border_color

    # Convert back to PIL images
    mask = Image.fromarray(mask_np)
    border = Image.fromarray(border_np)

    # Combine the mask and the border
    mask_with_border = Image.alpha_composite(mask, border)

    return mask_with_border


def load_and_remove_bg(path, size) -> Tuple[PILImage, PILImage]:
    """
    Load an image from a file and remove its background.

    Args:
        path: The path to the image file.
        size: The target size for the output image.

    Returns:
        Tuple[PILImage, PILImage]: The image with the removed background and the mask.
    """

    image = Image.open(path)
    # image = resize_image(image, size)
    image = resize_and_pad_image(image, size)
    image, mask = replace_background(image, size=size)

    return image, mask


def remove_bg_from_all_images(folder: str) -> None:
    """
    Remove the background from all images in a folder.

    Args:
        folder (str): The path to the folder containing the images.
    """
    for image in os.listdir(f'{folder}'):
        print("Removing background from", image)
        img, mask = load_and_remove_bg(f"{folder}/{image}", (300, 300))
        img.save(f"{folder}/{image}")


if __name__ == '__main__':
    folder_path = '../predicting/test_images'
    remove_bg_from_all_images(folder_path)
