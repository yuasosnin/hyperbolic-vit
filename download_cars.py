import os
import urllib.request
import tarfile

URL = "http://ai.stanford.edu/~jkrause/car196/"
PATH = "data/CARS196/"


if __name__ == "__main__":
    urllib.request.urlretrieve(URL + "car_devkit.tgz", PATH + "car_devkit.tgz")
    urllib.request.urlretrieve(URL + "cars_train.tgz", PATH + "cars_train.tgz")
    urllib.request.urlretrieve(URL + "cars_test.tgz",  PATH + "cars_test.tgz")
    urllib.request.urlretrieve(
        URL + "cars_test_annos_withlabels.mat", 
        PATH + "cars_test_annos_withlabels.mat")

    with tarfile.open(PATH + "car_devkit.tgz") as tar:
        tar.extractall(PATH)
    with tarfile.open(PATH + "cars_train.tgz") as tar:
        tar.extractall(PATH)
    with tarfile.open(PATH + "cars_test.tgz") as tar:
        tar.extractall(PATH)

    os.remove(PATH + "car_devkit.tgz")
    os.remove(PATH + "cars_train.tgz")
    os.remove(PATH + "cars_test.tgz")
