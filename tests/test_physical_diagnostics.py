import numpy as np
import pandas as pd
from core.physical_diagnostics import add_physical_diagnostic_features
def test_physical_diagnostics_are_finite_and_nonconstant():
    rows=[]
    for hour in range(2):
        for lat in range(2):
            for lon in range(2): rows.append({"datetime_utc":pd.Timestamp("2020-01-01")+pd.Timedelta(hours=hour),"V05002":lat,"V05001":lon,"T":20+lat,"P":1000,"RH":60,"PWV":hour+lat+2*lon,"U":1+lon,"V":1+lat,"dPWV+":7,"dPWV-":9,"V13011":0})
    out=add_physical_diagnostic_features(pd.DataFrame(rows)); cols=["specific_humidity_proxy","pwv_grad_mag","wind_convergence","moisture_advection"]; assert np.isfinite(out[cols].to_numpy()).all() and out.pwv_grad_mag.std()>0
