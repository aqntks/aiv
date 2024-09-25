
from typing import Any, Tuple
from PIL import ImageDraw, ImageFont

def normalize_coordinates(image_width: int, image_height: int, x: float, y: float, w: float, h: float, range: int = 100) -> Tuple[float, float, float, float]:
    """
    이미지 크기에 따라 좌표를 정규화합니다.

    Args:
        image_width (int): 이미지의 가로 크기.
        image_height (int): 이미지의 세로 크기.
        x (float): 정규화할 사각형의 좌상단 x 좌표.
        y (float): 정규화할 사각형의 좌상단 y 좌표.
        w (float): 정규화할 사각형의 너비.
        h (float): 정규화할 사각형의 높이.
        range (int, optional): 정규화 범위 (기본값은 100).

    Returns:
        Tuple[float, float, float, float]: 정규화된 (x, y, w, h) 좌표.
    """

    normalized_x = (x / image_width) * range
    normalized_y = (y / image_height) * range
    normalized_w = (w / image_width) * range
    normalized_h = (h / image_height) * range
    return normalized_x, normalized_y, normalized_w, normalized_h


def denormalize_coordinates(image_width: int, image_height: int, normalized_x: float, normalized_y: float, normalized_w: float, normalized_h: float, range: int = 100) -> Tuple[float, float, float, float]:
    """
    정규화된 좌표를 이미지 크기에 맞게 비정규화합니다.

    Args:
        image_width (int): 이미지의 가로 크기.
        image_height (int): 이미지의 세로 크기.
        normalized_x (float): 비정규화할 사각형의 좌상단 x 좌표.
        normalized_y (float): 비정규화할 사각형의 좌상단 y 좌표.
        normalized_w (float): 비정규화할 사각형의 너비.
        normalized_h (float): 비정규화할 사각형의 높이.
        range (int, optional): 정규화된 범위 (기본값은 100).

    Returns:
        Tuple[float, float, float, float]: 비정규화된 (x, y, w, h) 좌표.
    """

    x = (normalized_x / range) * image_width
    y = (normalized_y / range) * image_height
    w = (normalized_w / range) * image_width
    h = (normalized_h / range) * image_height
    return x, y, w, h


def visualization(pil_image: Any, text: str, box: Tuple[float, float, float, float], font: str) -> Any:
    """
    이미지에 텍스트와 사각형을 시각화합니다.

    Args:
        pil_image (Any): 텍스트와 사각형을 추가할 PIL 이미지 객체.
        text (str): 이미지에 그릴 텍스트.
        box (Tuple[float, float, float, float]): 사각형의 좌표 (x, y, w, h).
        font (str): 텍스트를 그릴 폰트.

    Returns:
        Any: 텍스트와 사각형이 추가된 PIL 이미지 객체.
    """

    draw = ImageDraw.Draw(pil_image)

    x, y, w, h = box
    x, y, w, h = float(x), float(y), float(w), float(h)
    text_position = (x - 10, y - 10)

    draw.rectangle([(x, y), (x + w, y + h)], outline="blue", width=3)

    try:
        font = ImageFont.truetype(font, 20) 
    except IOError:
        font = ImageFont.load_default(size=20)

    draw.text(text_position, text, fill="red", font=font)

    return pil_image
