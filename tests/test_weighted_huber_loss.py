import numpy as np
import pytest
tf = pytest.importorskip("tensorflow")
from core.models import grid_level_intensity_weighted_huber_loss
def test_weighted_huber_uses_formal_intensity_weights():
    truth=tf.constant([[0.,2.5,7.5,10.]],dtype=tf.float32); pred=tf.constant([[1.,3.5,8.5,11.]],dtype=tf.float32)
    value=float(grid_level_intensity_weighted_huber_loss(truth,pred).numpy()[0]); assert np.isclose(value,(.5*1+.5*2+.5*4+.5*8)/(1+2+4+8))
