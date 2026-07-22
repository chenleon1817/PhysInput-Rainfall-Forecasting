import pandas as pd
from core.data_processing import add_standard_and_derived_features
def test_dpwv_csv_columns_are_preserved():
    data=add_standard_and_derived_features(pd.DataFrame({"V13013":[7.25],"V13014":[3.5]})); assert data.loc[0,"dPWV+"]==7.25 and data.loc[0,"dPWV-"]==3.5
