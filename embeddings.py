import os
import shutil

import cv2
import numpy as np
import torch
import wget
from segment_anything import SamAutomaticMaskGenerator, SamPredictor, sam_model_registry
from tqdm import tqdm

import data
from models import imagebind_model
from models.imagebind_model import ModalityType


def get_masks(image_path, sam_checkpoint="sam_vit_b_01ec64.pth"):
    "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if not os.path.exists(f".checkpoints/{sam_checkpoint}"):
        print(f"Downloading sam weights to .checkpoints/{sam_checkpoint} ...")
        os.makedirs(".checkpoints", exist_ok=True)
        wget.download(
            f"https://dl.fbaipublicfiles.com/segment_anything/{sam_checkpoint}",
            f".checkpoints/{sam_checkpoint}",
        )

    if "vit_h" in sam_checkpoint:
        model_type = "vit_h"
    elif "vit_l" in sam_checkpoint:
        model_type = "vit_l"
    elif "vit_b" in sam_checkpoint:
        model_type = "vit_b"
    else:
        raise NameError(
            "There is no pretrained weight to download for %s, you need to provide a path to sam weights in ['sam_vit_h_4b8939.pth', 'sam_vit_l_0b3195.pth', 'sam_vit_b_01ec64.pth']."
            % sam_checkpoint
        )
    sam = sam_model_registry[model_type](checkpoint=f".checkpoints/{sam_checkpoint}")
    sam.to(device=device)

    mask_generator = SamAutomaticMaskGenerator(sam)

    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    masks = mask_generator.generate(image)
    sorted_masks = sorted(masks, key=(lambda x: x["area"]), reverse=True)
    masked_images = []
    for mask in sorted_masks:
        masked_image = np.zeros_like(image)
        masked_image[mask["segmentation"]] = image[mask["segmentation"]]
        masked_images.append(cv2.cvtColor(masked_image, cv2.COLOR_RGB2BGR))

    return sorted_masks, masked_images


def write_dicts_to_file(dict_list, file_path):
    with open(file_path, "w") as f:
        print("Filename:", dict_list, file=f)


def get_embeddings(filepath="./labels.txt", device="cuda:1"):
    colors = ["white", "yellow", "red", "blue", "green"]
    texts = [f"A {color} traffic sign." for color in colors]
    text_list = texts
    lines = open(filepath, "r").read().splitlines()
    success_count = 0
    approx_success_count = 0
    total = len(lines)
    print("Loading model...")
    model = imagebind_model.imagebind_huge(pretrained=True)
    model.eval()
    model.to(device)
    os.makedirs("./crops", exist_ok=True)
    shutil.rmtree("./wrongs")
    os.makedirs("./wrongs", exist_ok=True)
    wrongs = []
    with tqdm(range(total)) as pbar:
        for i in range(total):
            img_path, truth = lines[i].split(" ")
            mask_path = img_path[:-4] + "_mask.png"
            mask = cv2.imread(mask_path, 0)
            x, y, w, h = cv2.boundingRect(mask)
            img = cv2.imread(img_path)
            crop = img[y : y + h, x : x + w, :]
            crop_filename = f"crops/{os.path.basename(img_path)}.jpg"
            cv2.imwrite(crop_filename, crop)
            inputs = {
                ModalityType.TEXT: data.load_and_transform_text(text_list, device),
                ModalityType.VISION: data.load_and_transform_vision_data(
                    [crop_filename], device
                ),
            }
            with torch.no_grad():
                embeddings = model(inputs)

            probs = torch.softmax(
                embeddings[ModalityType.VISION] @ embeddings[ModalityType.TEXT].T,
                dim=-1,
            )
            output = probs[0].cpu().numpy()
            best = np.argmax(
                output,
                axis=0,
            )
            approx_colors = [colors[i] for i in range(len(output)) if output[i] > 0.15]
            predicted_color = colors[best]

            wrong_filename = f"wrongs/{os.path.basename(img_path)}.jpg"
            if predicted_color == truth:
                success_count += 1
                approx_success_count += 1
                pbar.set_description("Got it right.")
            elif predicted_color in approx_colors:
                approx_success_count += 1
                pbar.set_description("Got it approximately.")
                wrongs.append(
                    {
                        "image_path": crop_filename,
                        "predicted": predicted_color,
                        "real_color": truth,
                        "raw_output": output,
                    }
                )

                cv2.imwrite(wrong_filename, crop)
            else:
                pbar.set_description("Got it wrong.")
                wrongs.append(
                    {
                        "image_path": crop_filename,
                        "predicted": predicted_color,
                        "real_color": truth,
                        "raw_output": output,
                    }
                )

                cv2.imwrite(wrong_filename, crop)
            pbar.update(1)
    print("Real success", success_count / total)
    print("Approximate success:", approx_success_count / total)
    write_dicts_to_file(wrongs, "wrongs.txt")

    """
    output = probs[0].cpu().numpy()
    best = np.argmax(
        output,
        axis=0,
    )
    for i in range(len(output)):
        if output[i] > 0.15:
            print(mask_paths[i])
    print(f"Best mask: {mask_paths[best]}")
    """


if __name__ == "__main__":
    """
    use_masks = False

    if use_masks:
        if not os.path.exists("./masks/"):
            is_masks = False
            os.makedirs("./masks", exist_ok=True)
        else:
            is_masks = True
        if not is_masks:
            masks, masked_images = get_masks(image_path=".assets/car_image.jpg")
            mask_filenames = []
            for i, masked_image in enumerate(masked_images):
                filename = f"./masks/mask_{i+1:03d}.jpg"
                cv2.imwrite(filename, masked_image)
                mask_filenames.append(filename)
        else:
            mask_filenames = ["./masks/" + f for f in os.listdir("./masks/")]
        print(f"Using {len(os.listdir('./masks/'))} masks.")

        mask_paths = sorted(mask_filenames)

        images_paths = mask_paths
    else:
        images_paths = ["./images/" + path for path in sorted(os.listdir("./images/"))]
    """

    device = "cuda:1" if torch.cuda.is_available() else "cpu"
    get_embeddings(device=device)
