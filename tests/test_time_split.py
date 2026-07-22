import numpy as np
import pandas as pd
from core.data_processing import split_sequences_by_target_time
def test_fixed_split_boundaries():
    times=pd.to_datetime(["2024-06-30 23:59:59","2024-07-01 00:00:00","2024-12-31 23:59:59","2025-01-01 00:00:00"]); pack={"valid_times":times,"X_seq":np.zeros((4,1)),"y_reg":np.zeros((4,1)),"rain_seq_reg":np.zeros((4,1,1)),"grid_points_order":[],"feature_columns":[]}; split=split_sequences_by_target_time(pack); assert len(split["train"]["valid_times"])==1 and len(split["validation"]["valid_times"])==2 and len(split["test"]["valid_times"])==1
