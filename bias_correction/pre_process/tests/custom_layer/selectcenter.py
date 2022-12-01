import numpy as np
import matplotlib.pyplot as plt
import pytest

from bias_correction.train.model import SelectCenter
from bias_correction.tests.fixtures import image_cross


@pytest.fixture
def selectcenter():
    return SelectCenter(len_y=79, len_x=69)

