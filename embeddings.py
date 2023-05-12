import numpy as np
import torch

import data
from models import imagebind_model
from models.imagebind_model import ModalityType


def get_embeddings(texts: [str], masks: [np.ndarray]):
    device = "cuda:1" if torch.cuda.is_available() else "cpu"
    text_list = ["A dog"]
    image_paths = [
        ".assets/dog_image.jpg",
        ".assets/car_image.jpg",
        ".assets/bird_image.jpg",
    ]
    model = imagebind_model.imagebind_huge(pretrained=True)
    model.eval()
    model.to(device)
    inputs = {
        ModalityType.TEXT: data.load_and_transform_text(text_list, device),
        ModalityType.VISION: data.load_and_transform_vision_data(image_paths, device),
    }
    with torch.no_grad():
        embeddings = model(inputs)

    print(
        "Text x Vision: ",
        torch.softmax(
            embeddings[ModalityType.TEXT] @ embeddings[ModalityType.VISION].T, dim=-1
        ),
    )
    best = np.argmax(
        torch.softmax(
            embeddings[ModalityType.TEXT] @ embeddings[ModalityType.VISION].T, dim=-1
        )[0]
        .cpu()
        .numpy(),
        axis=0,
    )
    print(f"Image {image_paths[best]}")


if __name__ == "__main__":
    get_embeddings([""], [np.ndarray([0])])
