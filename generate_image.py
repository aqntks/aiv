
import os
import json
import random
from PIL import Image
from typing import List, Tuple, Dict

# config
max_length = 5
generate_size = (640, 640)
train_valid_test = (1000, 200, 100)
vocab = "15IPTUVY"
use_file_background = True

cropped_image_dir = "../OCR_dataset"
background_images_path = [
    "assets/background_src.png"
]  
output_dir = "dataset/aivocr"


def generate_random_string() -> str:
    """
    랜덤한 문자열을 생성합니다. 문자열의 길이는 1부터 max_length까지의 랜덤 값이며,
    vocab에 정의된 문자들로 구성됩니다.

    Returns:
        str: 생성된 랜덤 문자열
    """

    length = random.randint(1, max_length)
    letters_and_digits = vocab
    return ''.join(random.choice(letters_and_digits) for _ in range(length))


def get_cropped_image_for_char(char: str) -> Image.Image:
    """
    주어진 문자에 해당하는 랜덤한 BMP 이미지를 선택하여 반환합니다.

    Args:
        char (str): 문자 (문자에 해당하는 이미지 폴더 이름)

    Returns:
        Image.Image: 선택된 BMP 이미지
    """

    image_folder_path = os.path.join(cropped_image_dir, char)
    img_list = os.listdir(image_folder_path)
    img_list = [i for i in img_list if i.split(".")[-1].lower() == "bmp"]
    image_path = os.path.join(image_folder_path, img_list[random.randint(0, len(img_list) - 1)])
    return Image.open(image_path)


def create_random_background_image(width: int, height: int, base_color: tuple = (63, 65, 68)) -> Image.Image:
    """
    랜덤한 색상을 가진 배경 이미지를 생성합니다.

    Args:
        width (int): 이미지의 너비
        height (int): 이미지의 높이

    Returns:
        Image.Image: 생성된 배경 이미지
    """

    def vary_color(color: Tuple[int, int, int], variation: int = 10) -> Tuple[int, int, int]:
        return tuple(min(max(c + random.randint(-variation, variation), 0), 255) for c in color)
    
    background_color = vary_color(base_color, variation=10)
    background = Image.new("RGB", (width, height), background_color)
    
    return background


def get_background_from_file(width: int, height: int) -> Image.Image:
    """
    파일에서 배경 이미지를 불러오고, 필요시 지정된 크기로 리사이즈합니다.

    Args:
        width (int): 이미지의 너비
        height (int): 이미지의 높이

    Returns:
        Image.Image: 불러온 배경 이미지
    """
    
    image_path = random.choice(background_images_path)
    background = Image.open(image_path)
    
    if background.size != (width, height):
        background = background.resize((width, height))
    
    return background


def select_background(width: int, height: int, use_file_background: bool = False) -> Image.Image:
    """
    배경 이미지를 랜덤으로 생성하거나 파일에서 불러옵니다.

    Args:
        width (int): 이미지의 너비
        height (int): 이미지의 높이
        use_file_background (bool): 파일에서 배경을 불러올지 여부 (True이면 파일에서 불러옴)

    Returns:
        Image.Image: 선택된 배경 이미지
    """

    if use_file_background:
        return get_background_from_file(width, height)
    else:
        base_color = (63, 65, 68) # 회색
        return create_random_background_image(width, height, base_color)


def create_composite_image_and_gt(use_file_background: bool = True) -> Tuple[Image.Image, List[Dict[str, int]]]:
    """
    랜덤한 문자열을 생성하고, 문자를 합성한 이미지를 만들며, 이와 관련된 ground truth (GT) 데이터를 생성합니다.

    Args:
        use_file_background (bool): 파일에서 배경을 불러올지 여부 (True이면 파일에서 불러옴)

    Returns:
        Tuple[Image.Image, List[Dict[str, int]]]: 합성된 이미지와 GT 데이터
    """

    random_string = generate_random_string()
    background = select_background(generate_size[0], generate_size[1], use_file_background) 

    composite_image = background.copy()

    gt_data = []

    current_box = {"x": None, "y": None, "width": 0, "height": 0, "text": ""}
    
    total_text_width = 0
    max_text_height = 0
    for char in random_string:
        cropped_image = get_cropped_image_for_char(char)
        total_text_width += cropped_image.width
        max_text_height = max(max_text_height, cropped_image.height)

    x_offset = random.randint(10, background.width - total_text_width - 10)
    y_offset = random.randint(10, background.height - max_text_height - 10)
    
    box_id = 1

    for char in random_string:
        cropped_image = get_cropped_image_for_char(char)
        
        if cropped_image.mode != 'RGBA':
            cropped_image = cropped_image.convert('RGBA')

        if current_box["x"] is None:
            current_box["x"] = x_offset
            current_box["y"] = y_offset
        
        composite_image.paste(cropped_image, (x_offset, y_offset), cropped_image)

        current_box["width"] += cropped_image.width
        current_box["height"] = max(current_box["height"], cropped_image.height)
        current_box["text"] += char

        x_offset += cropped_image.width

    if current_box["text"]:
        gt_data.append({
            "text": current_box["text"],
            "box_id": box_id,
            "x": current_box["x"],
            "y": current_box["y"],
            "width": current_box["width"],
            "height": current_box["height"]
        })

    return composite_image, gt_data


def save_gt_to_json(all_gt_data: Dict[str, Dict[str, List]], json_file_path: str) -> None:
    """
    GT 데이터를 JSON 파일로 저장합니다.

    Args:
        all_gt_data (Dict[str, Dict[str, List]]): 저장할 GT 데이터
        json_file_path (str): 저장할 파일 경로
    """

    with open(json_file_path, 'w') as json_file:
        json.dump(all_gt_data, json_file, indent=4)


def save_as_jsonl(data_list: List[Dict[str, any]], file_path: str) -> None:
    """
    JSONL 형식으로 데이터를 저장합니다.

    Args:
        data_list (List[Dict[str, any]]): 저장할 데이터 리스트
        file_path (str): 저장할 파일 경로
    """
    
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data_list:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


if __name__ == "__main__":
    os.makedirs(output_dir, exist_ok=True)

    all_gt_data = dict()
    gt_jsonl = []

    for data_type in ["train", "validation", "test"]:
        os.makedirs(f"{output_dir}/{data_type}", exist_ok=False)

        if data_type == "train":
            count = train_valid_test[0]
        elif data_type == "validation":
            count = train_valid_test[1]
        elif data_type == "test":
            count = train_valid_test[2]
        else:
            raise NotImplementedError

        for i in range(1, count + 1):
            composite_image, gt_data = create_composite_image_and_gt(use_file_background=use_file_background)
            
            gt = gt_data[0]
            gt_parse = {
                "gt_parse": {
                    "ocr": gt["text"],
                    "box": [gt["x"], gt["y"], gt["width"], gt["height"]]
                }
            }
            gt_json = {
                "file_name": f"image_{i}.png", 
                "ground_truth": gt_parse
            }

            gt_jsonl.append(gt_json)
            
            composite_image.save(f"{output_dir}/{data_type}/image_{i}.png")

        save_as_jsonl(gt_jsonl, f'{output_dir}/{data_type}/metadata.jsonl')
