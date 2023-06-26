import argparse
import os


def folder_to_txt(folder, txtfile):
    """
    Given a folder path and a .txt file path, this function retrieves the absolute paths of all files in the folder
    and writes them to the .txt file, one path per line.
    Args:
        folder (str): The path to the folder.
        txtfile (str): The path to the .txt file.
    """
    with open(txtfile, "w") as f:
        for root, dirs, files in os.walk(folder):
            for file in sorted(files):
                f.write(os.path.join(root, file) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder_in", type=str, required=True)
    parser.add_argument("--textfile_out", type=str, required=True)
    args = parser.parse_args()

    folder_to_txt(args.folder_in, args.textfile_out)
