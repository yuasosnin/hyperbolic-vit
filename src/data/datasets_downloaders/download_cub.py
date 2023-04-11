from pathlib import Path
import os
import urllib.request
import tarfile

URL = "https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz?download=1"


def download_cub(root_dir='data'):
    root_dir = Path(root_dir)

    urllib.request.urlretrieve(URL, root_dir / "CUB_200_2011.tgz")
    with tarfile.open(root_dir / "CUB_200_2011.tgz") as tar:
        tar.extractall(root_dir)
    with tarfile.open(root_dir / "CUB_200_2011.tar") as tar:
        tar.extractall(root_dir)
    os.remove(root_dir / "CUB_200_2011.tgz")
    os.remove(root_dir / "CUB_200_2011.tar")
    os.remove(root_dir / "attributes.txt")


if __name__ == "__main__":
    download_cub()
