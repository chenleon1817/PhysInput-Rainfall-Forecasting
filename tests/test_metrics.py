import numpy as np
import pytest
from core.evaluation_utils import compute_continuous_metrics,compute_event_metrics
def test_metrics_and_shape_rejection():
    values=compute_continuous_metrics([1,2],[1,4]); assert values["MAE"]==1 and values["RMSE"]==2**.5
    events=compute_event_metrics([0,10],[0,10],[.1,2.5,10]); assert events[10.0]["hits"]==1
    with pytest.raises(ValueError): compute_continuous_metrics([1],[1,2])
