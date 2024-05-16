import base64
import io
from PIL import Image
from pathlib import Path


EXT_TO_MIMETYPE = {
    ".jpg": "image/jpeg",
    ".png": "image/png",
    ".svg": "image/svg+xml",
}


def base64_encode_image(image_path: str) -> str:
    image = Image.open(image_path)
    byte_arr = io.BytesIO()
    image.save(byte_arr, format="PNG")
    return base64.b64encode(byte_arr.getvalue()).decode("ascii")


def image_to_data_url(file_path):
    ext = Path(file_path).suffix  # Maybe introduce a mimetype map
    mimetype = EXT_TO_MIMETYPE[ext]
    with open(file_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode("utf-8")

    data_url = f"data:{mimetype};base64,{encoded_string}"
    return data_url
