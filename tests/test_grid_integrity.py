import pandas as pd
import pytest
from core.data_processing import validate_complete_grid
def test_grid_integrity():
    rows=[{"datetime_utc":"2020-01-01", "V05002":lat,"V05001":lon} for lat in range(9) for lon in range(5)]
    assert len(validate_complete_grid(pd.DataFrame(rows)))==45
    with pytest.raises(ValueError): validate_complete_grid(pd.DataFrame(rows[:-1]))
