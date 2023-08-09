import PIL.Image

from typing import Tuple
from PIL import ImageDraw


def draw_rectangle(
        orig: PIL.Image.Image,
        coords: Tuple[int, int, int, int],
        new_path: str,
        rect_color='red',
        rect_width=1) -> None:
    copy = orig.copy()
    draw = ImageDraw.Draw(copy)

    draw.rectangle(coords, outline=rect_color, width=rect_width)
    copy.save(new_path)
