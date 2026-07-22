import numpy as np
import pandas as pd
from core.data_processing import prepare_regional_grid_sequences
def test_sequence_target_alignment():
    rows=[]
    for hour in range(14):
        for lat in range(2):
            for lon in range(2): rows.append({"datetime_utc":pd.Timestamp("2020-01-01")+pd.Timedelta(hours=hour),"V05002":lat,"V05001":lon,"T":20.,"P":1000.,"RH":50.,"PWV":hour+lat+lon,"U":lon+1.,"V":lat+1.,"dPWV+":7,"dPWV-":9,"V13011":hour})
    pack=prepare_regional_grid_sequences(pd.DataFrame(rows),input_hours=12,grid_shape=(2,2),selected_features=["T","P","RH","PWV","U","V","DOY","HOD","dPWV+","dPWV-"])
    assert pack["X_seq"].shape[0]==2 and np.all(pack["y_reg"][0]==12) and pd.Timestamp(pack["valid_times"][0]).hour==12
