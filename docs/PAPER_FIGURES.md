# Paper Figures

`scripts/07_make_paper_figures.py` is the sole formal figure entry. It needs the event metrics table and long predictions for CNN-BiLSTM, PhysInput-Conv3, and PhysInput-All4. Heatmaps show Linear-ARX, BiLSTM, ConvLSTM, CNN-BiLSTM-Attention, CNN-BiLSTM, and PhysInput-All4. The rainfall-process figure uses fixed peak-day windows of 12:00–22:00 and 13:00–20:00 and writes PDF, SVG, and 600-dpi PNG files. Generated outputs remain separate from reported results.

It consumes evaluation outputs only; it never chooses a replacement event window.
