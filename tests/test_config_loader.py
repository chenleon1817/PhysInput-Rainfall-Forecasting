import pytest
from core.config_loader import *
def test_formal_configurations_are_exact():
    exp=load_experiment_config(); phys=load_physinput_model_config(); assert exp["models"]==FORMAL_MODEL_IDS; assert phys["shared_input_features"]==SHARED_FEATURES; assert phys["physical_diagnostic_features"]==DIAGNOSTIC_FEATURES; assert resolve_thresholds([.1,2.5,10])==[.1,2.5,10]
def test_invalid_configuration_is_rejected():
    data=load_physinput_model_config(); data["physinput_settings"]["fixed_physical_weights_used"]=True
    with pytest.raises(ValueError): validate_physinput_model_config(data)
