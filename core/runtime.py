"""Deterministic TensorFlow runtime helpers."""
from __future__ import annotations
import os
os.environ.setdefault("PYTHONHASHSEED", "42")
os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
os.environ.setdefault("TF_CUDNN_DETERMINISTIC", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import random
import numpy as np
import tensorflow as tf
_DEVICE_CONFIGURED=False
def set_deterministic_seed(seed):
    random.seed(seed); np.random.seed(seed); tf.keras.utils.set_random_seed(seed)
    if hasattr(tf.config.experimental, "enable_op_determinism"):
        tf.config.experimental.enable_op_determinism()

def reset_training_state(seed):
    tf.keras.backend.clear_session()
    set_deterministic_seed(seed)
def setup_device(prefer_gpu=True):
    global _DEVICE_CONFIGURED
    gpus=tf.config.list_physical_devices("GPU")
    if prefer_gpu and gpus and not _DEVICE_CONFIGURED:
        for gpu in gpus: tf.config.experimental.set_memory_growth(gpu,True)
        _DEVICE_CONFIGURED=True
    return "GPU" if prefer_gpu and gpus else "CPU"
def build_dataset(X,y,batch_size,seed,training=False):
    dataset=tf.data.Dataset.from_tensor_slices((X,y))
    if training: dataset=dataset.shuffle(min(len(X),10000),seed=int(seed),reshuffle_each_iteration=True)
    return dataset.batch(int(batch_size)).prefetch(tf.data.AUTOTUNE)
def fast_sliding_window(array,window):
    array=np.asarray(array)
    if window<=0 or len(array)<window: raise ValueError("window must be positive and no longer than the array.")
    shape=(len(array)-window+1,window)+array.shape[1:]; strides=(array.strides[0],)+array.strides
    return np.lib.stride_tricks.as_strided(array,shape=shape,strides=strides)
