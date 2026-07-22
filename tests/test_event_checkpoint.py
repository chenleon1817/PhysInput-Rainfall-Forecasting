import numpy as np
import pytest
tf = pytest.importorskip("tensorflow")
from core.training_utils import ValidationEventScoreCheckpointMetric
class _PredictionModel:
    def predict(self, values, batch_size, verbose): return np.asarray([[12.0]])
def test_event_checkpoint_populates_epoch_logs():
    callback=ValidationEventScoreCheckpointMetric(np.zeros((1,1)),np.asarray([[12.0]]),threshold=10.0); callback.set_model(_PredictionModel()); logs={}; callback.on_epoch_end(0,logs); assert {"val_event_score","val_csi_10","val_pod_10","val_far_10","val_fbias_10"}.issubset(logs)
