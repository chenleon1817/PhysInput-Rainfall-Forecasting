import numpy as np
import pytest
from core.data_processing import check_lagged_rainfall_no_target_leakage
def test_no_target_leakage_check():
    rain=np.zeros((2,2,45)); target=np.ones((2,45)); assert check_lagged_rainfall_no_target_leakage(rain,target)["status"]=="pass"
    with pytest.raises(ValueError): check_lagged_rainfall_no_target_leakage(np.ones((2,2,45)),target)
