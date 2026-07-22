import numpy as np
from core.result_export import export_predictions_long
def test_prediction_long_schema(tmp_path):
    output=export_predictions_long(tmp_path/"x.csv",["2020-01-01"],[(0,0)],np.array([[1.]]),np.array([[-1.]]),"bilstm",42); text=output.read_text(); assert "y_pred" in text and ",0.0," in text and "residual" in text
