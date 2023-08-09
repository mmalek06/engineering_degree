import PIL.Image


def resize_image(input_image_path, output_image_path, size):
    original_image = PIL.Image.open(input_image_path)
    width, height = original_image.size
    aspect_ratio = width / height

    if width > height:
        new_width = size[0]
        new_height = int(new_width / aspect_ratio)

        if new_height > size[1]:
            new_height = size[1]
            new_width = int(new_height * aspect_ratio)
    else:
        new_height = size[1]
        new_width = int(new_height * aspect_ratio)

        if new_width > size[0]:
            new_width = size[0]
            new_height = int(new_width / aspect_ratio)

    resized_img = original_image.resize((new_width, new_height))
    new_image = PIL.Image.new('RGB', size, 'black')

    new_image.paste(resized_img, ((size[0] - new_width) // 2,
                                  (size[1] - new_height) // 2))
    new_image.save(output_image_path)
