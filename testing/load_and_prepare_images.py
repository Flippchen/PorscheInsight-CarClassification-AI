import io
import os
from PIL import Image
from rembg import remove
from PIL.Image import Image as PILImage


def replace_background(im: bytes | PILImage, post_process_mask=False) -> PILImage:
    if not isinstance(im, PILImage):
        im = Image.open(io.BytesIO(im))

    im = remove(im, post_process_mask=post_process_mask)

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


def load_and_remove_bg(path, size):
    image = Image.open(path)
    image = resize_image(image, size)
    image = replace_background(image)

    return image


def remove_bg_from_all_images():
    for image in os.listdir('test_pic'):
        print("Removing background from", image)
        img = load_and_remove_bg(f"test_pic/{image}", (300, 300))
        img.save(f"test_pic/{image}")


if __name__ == '__main__':
    remove_bg_from_all_images()
