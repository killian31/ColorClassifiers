import os


def remove_masks(folder):
    """Removes every file that ends with 'mask.png' from a given folder.

    Args:
        folder (str): The path to the folder to be processed.

    Returns:
        None
    """

    for filename in os.listdir(folder):
        if filename.endswith("mask.png"):
            os.remove(os.path.join(folder, filename))


if __name__ == "__main__":
    remove_masks("./256_sampled_x150/")
