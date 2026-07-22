import numpy as np
import pytest
tf = pytest.importorskip("tensorflow")
from core.config_loader import load_experiment_config
from core.training_utils import build_intensity_sample_weight
def test_four_class_sample_weights():
    y=np.zeros((4,45)); y[1]=.2; y[2]=3; y[3]=12; weights,stats=build_intensity_sample_weight(y,load_experiment_config()); assert weights.tolist()==[1.,1.,1.5,2.] and stats["class_counts"]["heavy_rain"]==1
