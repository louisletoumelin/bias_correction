import numpy as np
import pytest


@pytest
def image_cross():
    image = np.zeros((1, 140, 140, 1))
    image[0, 40, :, 0] = 1
    image[0, 45, :, 0] = 1
    image[0, :, 100, 0] = 1
    return image

