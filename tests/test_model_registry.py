import pytest
tf = pytest.importorskip("tensorflow")
from core.config_loader import load_physinput_model_config
from core.models import get_formal_model_ids,get_required_diagnostics
def test_exact_formal_registry():
    ids=get_formal_model_ids(); assert len(ids)==9 and ids[0]=="linear_arx" and get_required_diagnostics("physinput_conv3",load_physinput_model_config())==["specific_humidity_proxy","pwv_grad_mag","wind_convergence"]
