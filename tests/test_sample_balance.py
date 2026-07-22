import numpy as np
import pandas as pd
from core.data_processing import build_event_aware_five_class_keep_indices,apply_training_sample_balance
def test_training_only_sampling_retains_protected_events(tmp_path):
    y=np.zeros((8,45)); y[3]=8; y[5]=12; times=pd.date_range("2024-01-01",periods=8,freq="h"); indices,_=build_event_aware_five_class_keep_indices(y,times,keep_ratios={"no_rain":.1,"light":1,"moderate":1,"near_heavy":1,"heavy":1,"event_neighbor":1}); assert {3,4,5,6}.issubset(set(indices)) and len(indices)==len(set(indices))
    split={"X_seq":np.zeros((8,1,9,5,1)),"y_reg":y,"rain_seq_reg":np.zeros((8,1,45)),"valid_times":times}; config={"rain_threshold_mm":.1,"moderate_threshold_mm":2.5,"near_heavy_threshold_mm":7.5,"heavy_threshold_mm":10.,"event_neighbor_hours":3,"keep_ratios":{"no_rain":.1,"light":1,"moderate":1,"near_heavy":1,"heavy":1,"event_neighbor":1},"seed":42}; balanced,_=apply_training_sample_balance(split,config,tmp_path); assert len(balanced["y_reg"])<=len(y) and (tmp_path/"sample_balance_manifest.json").is_file()
