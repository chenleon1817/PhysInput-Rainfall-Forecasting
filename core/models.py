"""Canonical model registry and builders for regional rainfall prediction."""
from __future__ import annotations
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from sklearn.linear_model import ElasticNet
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from .config_loader import FORMAL_MODEL_IDS, resolve_feature_columns

MODEL_DISPLAY_NAMES = {"linear_arx": "Linear-ARX", "bilstm": "BiLSTM", "convlstm": "ConvLSTM", "cnn_bilstm_attention": "CNN-BiLSTM-Attention", "cnn_bilstm": "CNN-BiLSTM", "physinput_lite": "PhysInput-Lite", "physinput_adv3": "PhysInput-Adv3", "physinput_conv3": "PhysInput-Conv3", "physinput_all4": "PhysInput-All4"}
DEEP_MODEL_IDS = tuple(model_id for model_id in FORMAL_MODEL_IDS if model_id != "linear_arx")
EXPECTED_INPUT_CHANNELS = {"bilstm": 11, "convlstm": 11, "cnn_bilstm_attention": 11, "cnn_bilstm": 11, "physinput_lite": 13, "physinput_adv3": 14, "physinput_conv3": 14, "physinput_all4": 15}

def get_formal_model_ids(): return list(FORMAL_MODEL_IDS)
def get_model_display_name(model_id):
    if model_id not in MODEL_DISPLAY_NAMES: raise ValueError(f"Unknown formal model ID: {model_id}")
    return MODEL_DISPLAY_NAMES[model_id]
def get_required_diagnostics(model_id, physinput_config): return resolve_feature_columns(model_id, physinput_config)[11:]

def grid_level_intensity_weighted_huber_loss(y_true, y_pred):
    """Return per-sample Huber loss with grid-cell target-intensity weights."""
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    error = y_true - y_pred
    abs_error = tf.abs(error)
    delta = tf.constant(1.0, dtype=tf.float32)
    huber = tf.where(abs_error <= delta, 0.5 * tf.square(error), delta * (abs_error - 0.5 * delta))
    weights = tf.ones_like(y_true, dtype=tf.float32)
    weights = tf.where(y_true >= 2.5, 2.0, weights)
    weights = tf.where(y_true >= 7.5, 4.0, weights)
    weights = tf.where(y_true >= 10.0, 8.0, weights)
    return tf.reduce_sum(weights * huber, axis=-1) / (tf.reduce_sum(weights, axis=-1) + tf.keras.backend.epsilon())

CUSTOM_OBJECTS = {"grid_level_intensity_weighted_huber_loss": grid_level_intensity_weighted_huber_loss}

def get_regional_loss(loss_mode):
    """Resolve the configured regional rainfall regression loss."""
    normalized = str(loss_mode or "grid_weighted_huber").lower()
    if normalized == "grid_weighted_huber": return grid_level_intensity_weighted_huber_loss
    if normalized == "plain_huber": return tf.keras.losses.Huber(delta=1.0)
    if normalized in {"mse", "mean_squared_error"}: return "mse"
    if normalized in {"mae", "mean_absolute_error"}: return "mae"
    raise ValueError(f"Unsupported loss mode: {loss_mode}. Expected grid_weighted_huber, plain_huber, mse, or mae.")
def _compile_model(model, config):
    training = config["training"] if "training" in config else config
    model.compile(optimizer=optimizers.Adam(learning_rate=float(training["learning_rate"]), clipnorm=float(training.get("optimizer_clipnorm", 1.0))), loss=get_regional_loss(training.get("loss_mode", "grid_weighted_huber")), metrics=["mae"])
    return model

def _temporal_attention_pool(x):
    scores = layers.Softmax(axis=1, name="temporal_attention_softmax")(layers.Dense(1, name="temporal_attention_score")(x))
    return layers.Lambda(lambda values: tf.reduce_sum(values[0] * values[1], axis=1), name="temporal_attention_weighted_sum")([x, scores])
def _build_grid_preserving_cnn_stem(inputs, filters, name="grid_cnn_stem"):
    """Build the formal six-branch stem in its original channel order."""
    local = layers.TimeDistributed(
        layers.Conv2D(filters, (3, 3), padding="same", activation="relu"),
        name=f"{name}_local_3x3",
    )(inputs)
    dilated = layers.TimeDistributed(
        layers.Conv2D(filters, (3, 3), dilation_rate=(2, 1), padding="same", activation="relu"),
        name=f"{name}_dilated_3x3",
    )(inputs)
    meridional = layers.TimeDistributed(
        layers.Conv2D(filters, (3, 1), padding="same", activation="relu"),
        name=f"{name}_meridional_3x1",
    )(inputs)
    zonal = layers.TimeDistributed(
        layers.Conv2D(filters, (1, 3), padding="same", activation="relu"),
        name=f"{name}_zonal_1x3",
    )(inputs)
    broad = layers.TimeDistributed(
        layers.Conv2D(filters, (5, 3), padding="same", activation="relu"),
        name=f"{name}_broad_5x3",
    )(inputs)
    raw_bypass = layers.TimeDistributed(
        layers.Conv2D(filters, (1, 1), padding="same", activation="relu"),
        name=f"{name}_raw_bypass",
    )(inputs)
    concatenated = layers.Concatenate(axis=-1, name=f"{name}_concat")(
        [local, dilated, meridional, zonal, broad, raw_bypass]
    )
    fused = layers.TimeDistributed(
        layers.Conv2D(filters * 2, (1, 1), padding="same", activation="relu"),
        name=f"{name}_fusion_1x1",
    )(concatenated)
    return layers.TimeDistributed(
        layers.BatchNormalization(), name=f"{name}_bn"
    )(fused)

def _grid_sequence_from_stem(stem, timesteps, channels):
    return layers.Lambda(lambda tensor: tf.reshape(tf.transpose(tensor, [0,2,3,1,4]), [-1, int(timesteps), channels]), output_shape=(int(timesteps), channels), name="per_grid_temporal_sequence")(stem)

def _restore_grid_context(temporal, inputs, height, width, units):
    return layers.Lambda(lambda values: tf.reshape(values[0], [tf.shape(values[1])[0], int(height), int(width), units * 2]), output_shape=(int(height), int(width), units * 2), name="restore_grid_context")([temporal, inputs])

def _build_post_temporal_spatial_refinement(x, filters, dropout_rate):
    residual = layers.Conv2D(filters, (1,1), padding="same", name="post_temporal_residual_1x1")(x)
    refined = layers.Conv2D(filters, (3,3), padding="same", activation="relu", name="post_temporal_refine_3x3")(x)
    refined = layers.Conv2D(filters, (3,1), padding="same", activation="relu", name="post_temporal_refine_3x1")(refined)
    refined = layers.Conv2D(filters, (1,3), padding="same", activation="relu", name="post_temporal_refine_1x3")(refined)
    refined = layers.BatchNormalization(name="post_temporal_refine_bn")(layers.Add(name="post_temporal_residual_add")([refined, residual]))
    return layers.Dropout(float(dropout_rate), name="post_temporal_refine_dropout")(refined)
def _grid_family(input_shape, config, attention=False, model_name="cnn_bilstm"):
    timesteps, height, width, _ = input_shape
    if None in (timesteps, height, width): raise ValueError("Grid model input dimensions must be fixed.")
    all_hparams = config["training"].get("model_hyperparameters", {}); hparams = all_hparams.get(model_name, all_hparams); units = int(hparams.get("lstm_units", hparams.get("attention_lstm_units", 64))); inputs = layers.Input(shape=input_shape, name="main_input")
    stem = _build_grid_preserving_cnn_stem(inputs, max(8, units // 2)); channels = max(8, units // 2) * 2
    sequence = _grid_sequence_from_stem(stem, timesteps, channels)
    temporal = layers.Bidirectional(layers.LSTM(units, return_sequences=attention), name="bilstm")(sequence)
    if attention: temporal = _temporal_attention_pool(temporal)
    restored = _restore_grid_context(temporal, inputs, height, width, units)
    restored = layers.Dropout(float(hparams.get("post_temporal_dropout", float(hparams.get("dropout_rate", .15)) * .5)), name="post_temporal_dropout")(restored)
    refined = _build_post_temporal_spatial_refinement(restored, max(32, units), hparams.get("dropout_rate", .15))
    return _compile_model(models.Model(inputs, layers.Flatten(name="rainfall_grid")(layers.Conv2D(1, (1,1), padding="same", name="rainfall_map_1x1")(refined)), name=model_name), config)
def create_bilstm_model(input_shape, config):
    all_hparams = config["training"].get("model_hyperparameters", {}); hparams = all_hparams.get("bilstm", all_hparams); units = int(hparams.get("lstm_units", hparams.get("attention_lstm_units", 64))); inputs=layers.Input(shape=input_shape, name="main_input"); x=layers.TimeDistributed(layers.Flatten(), name="td_flatten")(inputs); x=layers.Bidirectional(layers.LSTM(units), name="bilstm")(x); x=layers.Dropout(float(hparams.get("dropout_rate",.20)), name="dropout")(x); x=layers.Dense(128,activation="relu",name="dense_128")(x)
    return _compile_model(models.Model(inputs,layers.Dense(45,name="rainfall_grid")(x),name="bilstm"),config)
def build_convlstm_model(input_shape, config):
    all_hparams=config["training"].get("model_hyperparameters",{}); hparams=all_hparams.get("convlstm", all_hparams); inputs=layers.Input(shape=input_shape,name="main_input"); x=layers.ConvLSTM2D(32,(3,3),padding="same",name="convlstm_1")(inputs); x=layers.BatchNormalization(name="convlstm_bn")(x); x=layers.Conv2D(64,(3,3),padding="same",activation="relu",name="conv2d_post")(x); x=layers.GlobalAveragePooling2D(name="global_avg_pool")(x); x=layers.Dropout(float(hparams.get("dropout_rate",.20)),name="dropout")(x); x=layers.Dense(128,activation="relu",name="dense_128")(x)
    return _compile_model(models.Model(inputs,layers.Dense(45,name="rainfall_grid")(x),name="convlstm"),config)
def build_model(model_id, input_shape, config, physinput_config=None):
    if model_id not in DEEP_MODEL_IDS: raise ValueError(f"build_model supports only deep formal models; got {model_id}.")
    actual_channels = int(input_shape[-1]); expected_channels = EXPECTED_INPUT_CHANNELS[model_id]
    if actual_channels != expected_channels: raise ValueError(f"{model_id} requires {expected_channels} input channels, but received {actual_channels}.")
    if physinput_config is not None:
        expected_features = resolve_feature_columns(model_id, physinput_config)
        if len(expected_features) != expected_channels:
            raise ValueError(f"{model_id} configuration resolves {len(expected_features)} channels, expected {expected_channels}.")
    if model_id == "bilstm": return create_bilstm_model(tuple(input_shape), config)
    if model_id == "convlstm": return build_convlstm_model(tuple(input_shape), config)
    return _grid_family(tuple(input_shape), config, attention=model_id == "cnn_bilstm_attention", model_name=model_id)
def build_linear_arx_features(X_seq, rain_seq_reg):
    X, rain = np.asarray(X_seq,dtype=np.float32), np.asarray(rain_seq_reg,dtype=np.float32)
    if X.ndim != 5 or rain.ndim != 3 or X.shape[:2] != rain.shape[:2] or rain.shape[-1] != 45: raise ValueError("Linear-ARX inputs must be (N,T,H,W,F) and (N,T,45).")
    return np.concatenate([X.mean(axis=(2,3)).reshape(len(X),-1), X.std(axis=(2,3)).reshape(len(X),-1), X[:,-1].reshape(len(X),-1), rain.reshape(len(X),-1), rain.mean(axis=2), rain.max(axis=2)],axis=1)
def train_linear_arx_baseline(X_train,y_train,rain_seq_train,seed=42):
    model=Pipeline([("scaler",StandardScaler()),("regressor",MultiOutputRegressor(ElasticNet(alpha=.0005,l1_ratio=.15,max_iter=5000,random_state=int(seed),selection="cyclic"),n_jobs=-1))]); model.fit(build_linear_arx_features(X_train,rain_seq_train),np.asarray(y_train)); return model
def predict_linear_arx_baseline(model,X,rain_seq): return np.clip(np.asarray(model.predict(build_linear_arx_features(X,rain_seq)),dtype=np.float32),0,None)
