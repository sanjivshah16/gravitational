import streamlit as st
from pathlib import Path
import pandas as pd
from PIL import Image

st.set_page_config(layout="wide", page_title="Attention Mechanism Dashboard")
st.title("📊 Attention Mechanism Comparison Dashboard")

# Discover all result folders
result_dirs = sorted([d for d in Path(".").glob("results_*/analysis") if d.is_dir()], reverse=True)
selected_dir = st.selectbox("Select result folder:", result_dirs)

# Show summary table
summary_csv = selected_dir / "summary_metrics.csv"
if summary_csv.exists():
    df = pd.read_csv(summary_csv)
    st.subheader("📋 Summary Metrics")
    st.dataframe(df)
else:
    st.warning("summary_metrics.csv not found.")

# Show plots
def show_plot(filename, title):
    plot_path = selected_dir / filename
    if plot_path.exists():
        st.subheader(title)
        st.image(str(plot_path), use_column_width=True)
    else:
        st.info(f"{filename} not found.")

st.markdown("---")
col1, col2 = st.columns(2)

with col1:
    show_plot("accuracy_comparison.png", "🎯 Accuracy Comparison")
    show_plot("efficiency_comparison.png", "⚡ Efficiency Comparison")
    show_plot("convergence_comparison.png", "📉 Convergence Epochs")

with col2:
    show_plot("listops_learning_curves.png", "📈 ListOps Learning Curves")
    show_plot("listops_radar_comparison.png", "🌐 Radar Comparison")
    show_plot("listops_relative_improvement.png", "📊 Relative Improvement")

st.markdown("---")
show_plot("listops_attention_patterns.png", "🧠 Attention Patterns (ListOps)")

# Download option
st.markdown("---")
with open(selected_dir.parent.with_suffix(".zip"), "wb") as f:
    pass  # optionally zip the folder here later

