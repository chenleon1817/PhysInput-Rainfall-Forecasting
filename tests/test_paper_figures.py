import importlib.util
from pathlib import Path
import pandas as pd

def test_paper_figure_outputs(tmp_path):
    root=Path(__file__).resolve().parents[1]; source=root/"scripts"/"07_make_paper_figures.py"; spec=importlib.util.spec_from_file_location("paper_figures",source); module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    evaluation=tmp_path/"evaluation"; predictions=evaluation/"predictions_long"; predictions.mkdir(parents=True); models=["linear_arx","bilstm","convlstm","cnn_bilstm_attention","cnn_bilstm","physinput_all4"]
    events=[]
    for model in models:
        for threshold in [.1,2.5,10.]: events.append({"seed":42,"model_id":model,"threshold_mm":threshold,"POD":.5,"FAR":.2,"CSI":.4,"FBIAS":1.})
    pd.DataFrame(events).to_csv(evaluation/"formal_event_metrics_by_seed.csv",index=False)
    hours=pd.date_range("2025-01-01 12:00",periods=11,freq="h")
    for model in ["cnn_bilstm","physinput_conv3","physinput_all4"]:
            pd.DataFrame([{ "valid_time":time,"point_idx":0,"latitude":0,"longitude":0,"y_true":12 if index==4 else 1,"y_pred":10 if index==4 else 1,"seed":42} for index,time in enumerate(hours)]).to_csv(predictions/f"{model}_predictions_long.csv",index=False)
    output=tmp_path/"figures"; module.draw_event_metric_heatmaps(evaluation,output,42); module.draw_typical_rainfall_process(evaluation,output)
    for name in ["event_metric_heatmaps","typical_rainfall_process"]:
        for extension in ["pdf","svg","png"]: assert (output/extension/f"{name}.{extension}").stat().st_size>0
    assert (output/"selected_rainfall_window.json").is_file()
