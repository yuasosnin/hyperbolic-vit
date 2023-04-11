from pathlib import Path
import os
import urllib.request
import tarfile

URL = "http://ai.stanford.edu/~jkrause/car196/"
URL_DEVKIT = "https://ai.stanford.edu/~jkrause/cars/"


def download_cars(root_dir="data"):
    root_dir = Path(root_dir)
    data_dir = "CARS196"
    (root_dir / data_dir).mkdir(parents=True, exist_ok=True)

    urllib.request.urlretrieve(URL_DEVKIT + "car_devkit.tgz", root_dir / "car_devkit.tgz")
    urllib.request.urlretrieve(URL + "cars_train.tgz", root_dir / "cars_train.tgz")
    urllib.request.urlretrieve(URL + "cars_test.tgz",  root_dir / "cars_test.tgz")
    urllib.request.urlretrieve(
        URL + "cars_test_annos_withlabels.mat", 
        root_dir / data_dir / "cars_test_annos_withlabels.mat")

    with tarfile.open(root_dir / "car_devkit.tgz") as tar:
        tar.extractall(root_dir / data_dir)
    with tarfile.open(root_dir / "cars_train.tgz") as tar:
        tar.extractall(root_dir / data_dir)
    with tarfile.open(root_dir / "cars_test.tgz") as tar:
        tar.extractall(root_dir / data_dir)

    os.remove(root_dir / "car_devkit.tgz")
    os.remove(root_dir / "cars_train.tgz")
    os.remove(root_dir / "cars_test.tgz")


if __name__ == "__main__":
    download_cars()
