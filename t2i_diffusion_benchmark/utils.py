import base64
import io
import os
from PIL import Image


def pathify_image_url(image_url: str, generated_images_dir: str, huggingface_repo: str):
    return (
        os.path.join(generated_images_dir, image_url.split("/")[-1])
        if huggingface_repo is not None
        else image_url
    )


def base64_encode_image(image_path: str) -> str:
    image = Image.open(image_path)
    byte_arr = io.BytesIO()
    image.save(byte_arr, format="PNG")
    return base64.b64encode(byte_arr.getvalue()).decode("ascii")
