# 1. README.md – Project Overview

## Project Title
**Interactive Linear Regression Demo (Streamlit + CRISP-DM)**

## Description
This project is an educational web app demonstrating **linear regression** and **polynomial regression** modeling, following the **CRISP-DM framework**. It was built using **Streamlit**, with interactive controls, animations, and report generation. Designed for classroom use, student demos, or data science portfolios.

### Key Features
- **CRISP-DM Workflow Integration** — Business Understanding → Data Understanding → Data Preparation → Modeling → Evaluation → Deployment.
- **Interactive Controls** — Users can modify the slope `a`, intercept `b`, number of data points, and noise level.
- **Animated Visuals (Plotly)** — Dynamic polynomial regression fits and residual plots.
- **Compare Models Tab** — Auto-play through different polynomial degrees with an animated progress bar.
- **Model Insights Tab** — Residual animations for visual diagnostics.
- **PDF Report Generation** — Includes CRISP-DM explanations, progress timeline, and data summary using `reportlab`.
- **Educational Design** — Visual and interactive flow ideal for classroom learning or portfolio demonstration.

### Technologies Used
- **Python 3.10+**
- **Streamlit** — web UI framework
- **Plotly** — animated and interactive visualizations
- **Scikit-learn** — linear and polynomial regression models
- **ReportLab** — for PDF report generation
