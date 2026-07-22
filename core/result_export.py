"""Stable result-table and long-prediction export helpers."""
from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import pandas as pd
from .models import get_model_display_name

def save_prediction_npz(path, y_true, y_pred, valid_times, grid_points_order, model_id, seed):
    """Save a canonical clipped prediction archive."""
    true,pred=np.asarray(y_true),np.clip(np.asarray(y_pred),0,None)
    if true.shape != pred.shape: raise ValueError("Prediction archive arrays must have the same shape.")
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True); np.savez_compressed(path,y_true=true,y_pred=pred,valid_times=np.asarray(valid_times),grid_points_order=np.asarray(grid_points_order),model_id=str(model_id),seed=int(seed)); return path

def export_predictions_long(path, valid_times, grid_points_order, y_true, y_pred, model_id, seed):
    true,pred=np.asarray(y_true),np.asarray(y_pred)
    if true.shape != pred.shape or true.ndim != 2 or true.shape[1] != len(grid_points_order): raise ValueError("Long prediction export requires aligned (samples, cells) arrays.")
    rows=[]
    for index,time in enumerate(pd.to_datetime(valid_times)):
        for point,(latitude,longitude) in enumerate(grid_points_order):
            value=float(max(pred[index,point],0.0)); observed=float(true[index,point]); rows.append({"valid_time":time.isoformat(),"point_idx":point,"latitude":latitude,"longitude":longitude,"y_true":observed,"y_pred":value,"residual":value-observed,"model_id":model_id,"seed":int(seed)})
    output=Path(path); output.parent.mkdir(parents=True,exist_ok=True); pd.DataFrame(rows).to_csv(output,index=False); return output
def export_evaluation_tables(output_dir, metric_rows, event_rows, intensity_rows):
    output=Path(output_dir); output.mkdir(parents=True,exist_ok=True)
    metrics=pd.DataFrame(metric_rows); events=pd.DataFrame(event_rows); intensity=pd.DataFrame(intensity_rows)
    required={"seed","result_key","model_id","model_name","MAE","RMSE","R2","Bias","Correlation","MAE_valid_rain"}
    if not required.issubset(metrics): raise ValueError(f"Metric table lacks required columns: {sorted(required-set(metrics))}")
    required_events={"seed","model_id","model_name","threshold_mm","POD","FAR","CSI","FBIAS","hits","misses","false_alarms","correct_negatives","observed_event_count","predicted_event_count"}
    required_intensity={"seed","model_id","model_name","true_no_rain","pred_no_rain","y_true_max","y_pred_max"}
    if not required_events.issubset(events): raise ValueError(f"Event table lacks required columns: {sorted(required_events-set(events))}")
    if not required_intensity.issubset(intensity): raise ValueError(f"Intensity table lacks required columns: {sorted(required_intensity-set(intensity))}")
    metrics.to_csv(output / "formal_metrics_by_seed.csv",index=False,float_format="%.10g"); events.to_csv(output / "formal_event_metrics_by_seed.csv",index=False,float_format="%.10g"); intensity.to_csv(output / "prediction_intensity_distribution.csv",index=False,float_format="%.10g")
    def mean_population_std(frame, group_columns, value_columns):
        rows=[]
        for keys, group in frame.groupby(group_columns, dropna=False):
            keys=(keys,) if not isinstance(keys, tuple) else keys
            row=dict(zip(group_columns, keys))
            for column in value_columns:
                values=np.asarray(group[column],dtype=float)
                row[f"{column}_mean"]=float(np.nanmean(values))
                row[f"{column}_std"]=float(np.nanstd(values,ddof=0))
            rows.append(row)
        return pd.DataFrame(rows)
    mean_population_std(metrics,["model_id","model_name"],["MAE","RMSE","R2","Bias","Correlation","MAE_valid_rain"]).to_csv(output / "formal_metrics_mean_std.csv",index=False,float_format="%.10g")
    mean_population_std(events,["model_id","model_name","threshold_mm"],["POD","FAR","CSI","FBIAS"]).to_csv(output / "formal_event_metrics_mean_std.csv",index=False,float_format="%.10g")
    return output
def write_json(path, payload):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True); path.write_text(json.dumps(payload,indent=2,default=str),encoding="utf-8")

def export_model_metrics(path, metrics): write_json(path, metrics)
def export_model_history(path, history): pd.DataFrame(history).to_csv(Path(path),index=False)
def export_model_train_info(path, info): write_json(path, info)

def export_physinput_ablation_at10(path, event_rows, heavy_threshold=10.0):
    order=["cnn_bilstm","physinput_lite","physinput_adv3","physinput_conv3","physinput_all4"]
    table=pd.DataFrame(event_rows)
    columns=["seed","model_id","model_name","POD","FAR","CSI","FBIAS","hits","misses","false_alarms","observed_event_count","predicted_event_count"]
    result=table[np.isclose(table["threshold_mm"].astype(float),float(heavy_threshold)) & table["model_id"].isin(order)].copy()
    result["model_id"]=pd.Categorical(result["model_id"],categories=order,ordered=True)
    result=result.sort_values(["seed","model_id"])[columns]
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    result.to_csv(path,index=False,float_format="%.10g")
    return Path(path)
