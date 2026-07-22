#!/usr/bin/env python3
"""The sole public entry point for the two final paper figures."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HEATMAP_MODELS=[("linear_arx","Linear-ARX"),("bilstm","BiLSTM"),("convlstm","ConvLSTM"),("cnn_bilstm_attention","CNN-BiLSTM-Attention"),("cnn_bilstm","CNN-BiLSTM"),("physinput_all4","PhysInput-All4")]
PROCESS_MODELS=[("cnn_bilstm","CNN-BiLSTM","#377eb8"),("physinput_conv3","PhysInput-Conv3","#ff7f00"),("physinput_all4","PhysInput-All4","#e41a1c")]
THRESHOLDS=[.1,2.5,10.]
def set_paper_style(): mpl.rcParams.update({"font.family":"DejaVu Sans","pdf.fonttype":42,"ps.fonttype":42,"axes.linewidth":.7})
def _save(figure, output_dir, name):
    paths={};
    for extension in ("pdf","svg","png"):
        path=output_dir / extension / f"{name}.{extension}"; path.parent.mkdir(parents=True,exist_ok=True); figure.savefig(path,dpi=600 if extension=="png" else None,bbox_inches="tight",pad_inches=.02); paths[extension]=str(path.relative_to(output_dir))
    plt.close(figure); return paths
def draw_event_metric_heatmaps(evaluation_dir, output_dir, seed):
    table=pd.read_csv(Path(evaluation_dir)/"formal_event_metrics_by_seed.csv"); table=table[table["seed"].astype(int)==int(seed)]
    set_paper_style(); figure,axes=plt.subplots(1,4,figsize=(7.16,2.72),sharey=True)
    for axis,metric in zip(axes,["POD","FAR","CSI","FBIAS"]):
        matrix=np.full((len(HEATMAP_MODELS),3),np.nan)
        for row,(model_id,_) in enumerate(HEATMAP_MODELS):
            for col,threshold in enumerate(THRESHOLDS):
                values=table[(table.model_id==model_id)&np.isclose(table.threshold_mm,threshold)]
                if not values.empty: matrix[row,col]=float(values.iloc[0][metric])
        score=1-np.clip(matrix,0,1) if metric=="FAR" else 1-np.clip(abs(matrix-1)/.8,0,1) if metric=="FBIAS" else np.clip(matrix,0,1); axis.imshow(score,cmap="YlGnBu",aspect="auto",vmin=0,vmax=1); axis.set_title(metric if metric!="FAR" else "FAR (lower is better)",fontsize=7); axis.set_xticks(range(3),["0.1","2.5","10.0"],fontsize=6); axis.set_xlabel("Threshold (mm h$^{-1}$)",fontsize=6); axis.set_yticks(range(6),[name for _,name in HEATMAP_MODELS] if axis is axes[0] else [],fontsize=6)
        for r in range(6):
            for c in range(3): axis.text(c,r,"NA" if not np.isfinite(matrix[r,c]) else f"{matrix[r,c]:.3f}",ha="center",va="center",fontsize=5.6)
    figure.subplots_adjust(left=.18,right=.998,bottom=.22,top=.86,wspace=.06); return _save(figure,Path(output_dir),"event_metric_heatmaps")
def _domain_max(path, seed):
    data=pd.read_csv(path)
    if "seed" not in data: raise ValueError(f"Prediction table has no seed column: {path}")
    data=data[data["seed"].astype(int)==int(seed)].copy()
    if data.empty: raise ValueError(f"Prediction table has no rows for seed {seed}: {path}")
    data["valid_time"]=pd.to_datetime(data["valid_time"])
    return data.groupby("valid_time",as_index=False).agg(y_true=("y_true","max"),y_pred=("y_pred","max"),grid_cells=("point_idx","nunique")).sort_values("valid_time")
def draw_typical_rainfall_process(evaluation_dir,output_dir,seed=42):
    directory=Path(evaluation_dir)/"predictions_long"; series={model:_domain_max(directory/f"{model}_predictions_long.csv",seed) for model,_,_ in PROCESS_MODELS}; reference=series["cnn_bilstm"]; times=reference.valid_time.reset_index(drop=True); observed=reference.y_true.to_numpy(); peak=int(np.argmax(observed)); day=times.iloc[peak].normalize(); windows=[(12,22,"(a) Overall rainfall process"),(13,20,"(b) Heavy-rainfall zoom")]
    set_paper_style(); figure,axes=plt.subplots(2,1,figsize=(5.25,2.86)); selection={"peak_time":times.iloc[peak].isoformat(),"windows":[]}
    ymax=max(30,float(np.ceil(max(max(observed),*(frame.y_pred.max() for frame in series.values()))/5)*5))
    for axis,(start_hour,end_hour,title) in zip(axes,windows):
        mask=(times.dt.normalize()==day)&(times.dt.hour>=start_hour)&(times.dt.hour<=end_hour)
        if not mask.any(): raise ValueError(f"Required fixed clock window {start_hour}:00-{end_hour}:00 is unavailable on the observed peak day.")
        subset=times[mask]; x=np.arange(len(subset)); axis.plot(x,observed[mask],color="black",label="Observed",linewidth=1.3)
        for model,label,color in PROCESS_MODELS:
            frame=series[model]
            if not np.array_equal(frame.valid_time.to_numpy(),times.to_numpy()) or not np.array_equal(frame.y_true.to_numpy(),observed) or not np.array_equal(frame.grid_cells.to_numpy(),reference.grid_cells.to_numpy()): raise ValueError(f"Observed values, timestamps, or grid coverage do not match for {model}.")
            axis.plot(x,np.clip(frame.y_pred.to_numpy()[mask],0,None),label=label,color=color,linewidth=1.0)
        axis.axhline(10,color=".35",linestyle="--",linewidth=.6); axis.set_ylim(0,ymax); axis.set_title(title,fontsize=7,loc="left"); axis.set_ylabel("Rainfall (mm h$^{-1}$)",fontsize=6.8); axis.set_xticks(x[::max(1,len(x)//5)],subset.dt.strftime("%H:%M").iloc[::max(1,len(x)//5)],fontsize=6); axis.grid(axis="y",linestyle="--",alpha=.25,linewidth=.4)
        if times.iloc[peak] in set(subset):
            peak_position=int(np.where(subset.to_numpy()==times.iloc[peak].to_datetime64())[0][0]); axis.axvline(peak_position,color=".25",linestyle="--",linewidth=.7); axis.annotate(f"Observed peak\n{observed[peak]:.2f} mm h$^{{-1}}$",(peak_position,observed[peak]),xytext=(4,-4),textcoords="offset points",fontsize=5.8,va="top")
        selection["windows"].append({"panel":title,"start":subset.iloc[0].isoformat(),"end":subset.iloc[-1].isoformat()})
    axes[0].legend(ncol=4,fontsize=5.7,frameon=False,loc="upper left"); axes[1].set_xlabel(f"Valid time ({day.strftime('%m-%d')})",fontsize=6.8); figure.tight_layout(pad=.4); paths=_save(figure,Path(output_dir),"typical_rainfall_process"); Path(output_dir,"selected_rainfall_window.json").write_text(json.dumps(selection,indent=2),encoding="utf-8"); return paths
def parse_args():
    parser=argparse.ArgumentParser(description="Make the final event heatmap and rainfall-process figures."); parser.add_argument("--evaluation-dir",type=Path,required=True); parser.add_argument("--output-dir",type=Path,default=Path("outputs/paper_figures")); parser.add_argument("--seed",type=int,default=42); return parser.parse_args()
def main():
    args=parse_args(); heatmap=draw_event_metric_heatmaps(args.evaluation_dir,args.output_dir,args.seed); process=draw_typical_rainfall_process(args.evaluation_dir,args.output_dir,args.seed); Path(args.output_dir).mkdir(parents=True,exist_ok=True); manifest={"source_evaluation_directory":str(args.evaluation_dir),"seed":args.seed,"model_ids":[model for model,_,_ in PROCESS_MODELS],"thresholds":THRESHOLDS,"fixed_windows":["12:00-22:00","13:00-20:00"],"event_metric_heatmaps":heatmap,"typical_rainfall_process":process,"selected_rainfall_window":"selected_rainfall_window.json"}; Path(args.output_dir,"figure_manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
if __name__=="__main__": main()
