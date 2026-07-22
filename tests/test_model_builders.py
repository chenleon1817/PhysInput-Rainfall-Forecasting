import pytest
tf = pytest.importorskip("tensorflow")
from core.config_loader import load_experiment_config,load_physinput_model_config
from core.models import DEEP_MODEL_IDS,build_model
def test_deep_builders_output_45_cells():
    config=load_experiment_config(); phys=load_physinput_model_config()
    for model_id in DEEP_MODEL_IDS:
        features=11+len(phys["physinput_variants"].get(model_id,[])); model=build_model(model_id,(2,9,5,features),config,phys); assert model.output_shape==(None,45)
    tf.keras.backend.clear_session()
