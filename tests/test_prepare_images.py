import os
from PIL import Image
from utilities.prepare_images import (get_session, replace_background, get_bounding_box,
                                      resize_cutout, resize_image, resize_and_pad_image,
                                      fix_image, load_and_remove_bg, remove_bg_from_all_images)


# Make sure to replace 'your_module' with the actual name of your module file

def test_get_session():
    session = get_session()
    assert session is not None
    assert session.model_name == 'u2net'


def test_replace_background():
    # Test using a sample image
    im = Image.new('RGB', (100, 100), 'WHITE')
    new_im, mask = replace_background(im)
    assert new_im is not None
    assert isinstance(new_im, Image.Image)
    assert mask is not None
    assert isinstance(mask, Image.Image)


def test_get_bounding_box():
    im = Image.new('RGBA', (100, 100), 'WHITE')
    bbox = get_bounding_box(im)
    assert bbox == (0, 0, 99, 99)


def test_resize_cutout():
    im = Image.new('RGBA', (100, 100), 'WHITE')
    resized_im = resize_cutout(im, (50, 50))
    assert resized_im.size == (50, 50)


def test_resize_image():
    im = Image.new('RGBA', (100, 100), 'WHITE')
    resized_im = resize_image(im, (50, 50))
    assert resized_im.size == (50, 50)


def test_resize_and_pad_image():
    im = Image.new('RGBA', (100, 100), 'WHITE')
    resized_im = resize_and_pad_image(im, (50, 50))
    assert resized_im.size == (50, 50)


def test_fix_image():
    im = Image.new('RGBA', (100, 100), 'WHITE')
    fixed_im = fix_image(im)
    assert fixed_im.mode == 'RGB'


def test_load_and_remove_bg(tmpdir):
    test_image = Image.new('RGB', (100, 100), 'WHITE')
    test_image_path = os.path.join(tmpdir, 'test_image.jpg')
    test_image.save(test_image_path)
    loaded_im, mask = load_and_remove_bg(test_image_path, (50, 50))
    assert loaded_im is not None
    assert loaded_im.size == (50, 50)
    assert mask is not None
    assert mask.size == (50, 50)


def test_remove_bg_from_all_images(tmpdir):
    test_folder = os.path.join(tmpdir, 'test_images')
    os.makedirs(test_folder)
    for i in range(5):
        test_image = Image.new('RGB', (100, 100), 'WHITE')
        test_image_path = os.path.join(test_folder, f'test_image_{i}.jpg')
        test_image.save(test_image_path)

    remove_bg_from_all_images(test_folder)

    for i in range(5):
        assert os.path.exists(os.path.join(test_folder, f'test_image_{i}.jpg'))

