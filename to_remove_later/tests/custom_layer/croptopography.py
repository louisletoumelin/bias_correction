import numpy as np
import matplotlib.pyplot as plt
import pytest

from bias_correction.train.model import CropTopography


@pytest.fixture
def croptopography():
    return CropTopography(initial_length=140, y_offset=39, x_offset=34)


def figure_croptopography(image_cross, croptopography):
    plt.figure()
    plt.subplot(121)
    plt.imshow(image_cross[0, :, :, 0])
    plt.title("Before crop")
    plt.subplot(122)
    result = croptopography()[0, :, :, 0]
    plt.imshow(result)
    plt.title(f"After crop")
    plt.colorbar()

