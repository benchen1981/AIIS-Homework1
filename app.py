"""
Streamlit app: Interactive linear regression demo for classroom/portfolio
Features (as requested):
- Follows CRISP-DM steps with prompts & progress timeline
- User controls for true model a, b in ax+b, noise, number of points
- Tabs: Explore, Compare Models, Model Insights
- Animated Plotly charts + residuals
- Polynomial Degree animation slider with autoplay/loop/pause/resume/speed
- Download Report (PDF) using reportlab including CRISP-DM explanations and progress
- Compare Models autoplay with progress bar under the chart

Run: `streamlit run streamlit_linear_regression_app.py`

Dependencies: streamlit, numpy, pandas, scikit-learn, plotly, reportlab
Install: pip install streamlit numpy pandas scikit-learn plotly reportlab
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as graphobjects
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, r2_score
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from io import BytesIO
import time

# ---------- Utility functions ----------

def generate_data(a, b, n, noise_std, x_min=0, x_max=10, random_seed=None):
    rng = np.random.default_rng(random_seed)
    x = rng.uniform(x_min, x_max, size=n)
    y = a * x + b + rng.normal(0, noise_std, size=n)
    df = pd.DataFrame({"x": x, "y": y})
    return df.sort_values('x')


def fit_model(degree, df):
    X = df[["x"]].values
    y = df["y"].values
    if degree == 1:
        model = LinearRegression()
        model.fit(X, y)
        y_pred = model.predict(X)
        return model, y_pred
    else:
        model = make_pipeline(PolynomialFeatures(degree, include_bias=False), LinearRegression())
        model.fit(X, y)
        y_pred = model.predict(X)
        return model, y_pred


def compute_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return mse, r2


def create_crisp_dm_text(data_description, modeling_choice, evaluation_summary):
    steps = []
    steps.append(("Business Understanding", "Goal: teach/visualize linear regression using a simple synthetic dataset; allow students to change parameters and observe model behavior."))
    steps.append(("Data Understanding", data_description))
    steps.append(("Data Preparation", "We generated synthetic data with controllable slope/intercept/noise; split not necessary for demo but could be added."))
    steps.append(("Modeling", modeling_choice))
    steps.append(("Evaluation", evaluation_summary))
    steps.append(("Deployment", "This Streamlit app demonstrates deployment; PDF report generation uses reportlab and is downloadable."))
    return steps


def generate_pdf_report(buffer, crisp_dm_steps, prompt_text, progress_perc, df, model_info):
    # buffer: BytesIO
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter
    margin = 50
    y = height - margin
    c.setFont("Helvetica-Bold", 16)
    c.drawString(margin, y, "Linear Regression Demo Report")
    y -= 30
    c.setFont("Helvetica", 10)
    c.drawString(margin, y, f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    y -= 20

    c.setFont("Helvetica-Bold", 12)
    c.drawString(margin, y, "User Prompt & Inputs")
    y -= 16
    c.setFont("Helvetica", 10)
    for line in prompt_text.split('\n'):
        c.drawString(margin, y, line)
        y -= 12
        if y < margin:
            c.showPage(); y = height - margin

    c.setFont("Helvetica-Bold", 12)
    c.drawString(margin, y, "CRISP-DM Steps")
    y -= 14
    c.setFont("Helvetica", 10)
    for title, text in crisp_dm_steps:
        c.drawString(margin, y, f"{title}:")
        y -= 12
        for paragraph in text.split('\n'):
            c.drawString(margin + 10, y, paragraph)
            y -= 12
            if y < margin:
                c.showPage(); y = height - margin
        y -= 6

    c.setFont("Helvetica-Bold", 12)
    c.drawString(margin, y, "Progress Snapshot")
    y -= 14
    c.setFont("Helvetica", 10)
    c.drawString(margin, y, f"Timeline progress at time of report generation: {progress_perc}%")
    y -= 16

    c.setFont("Helvetica-Bold", 12)
    c.drawString(margin, y, "Dataset Sample")
    y -= 14
    c.setFont("Helvetica", 9)
    # write first 10 rows
    for i, row in df.head(10).iterrows():
        c.drawString(margin, y, f"x={row['x']:.3f}, y={row['y']:.3f}")
        y -= 10
        if y < margin:
            c.showPage(); y = height - margin

    c.setFont("Helvetica-Bold", 12)
    c.drawString(margin, y, "Model Summary")
    y -= 14
    c.setFont("Helvetica", 10)
    for k, v in model_info.items():
        c.drawString(margin, y, f"{k}: {v}")
        y -= 12
        if y < margin:
            c.showPage(); y = height - margin

    c.showPage()
    c.save()
    buffer.seek(0)
    return buffer

# ---------- Streamlit UI ----------

st.set_page_config(page_title="Linear Regression Classroom Demo", layout='wide')

# Initialize session state for animation controls
if 'play' not in st.session_state:
    st.session_state.play = False
if 'loop' not in st.session_state:
    st.session_state.loop = False
if 'paused' not in st.session_state:
    st.session_state.paused = False
if 'frame' not in st.session_state:
    st.session_state.frame = 0
if 'last_update' not in st.session_state:
    st.session_state.last_update = time.time()

# Sidebar: inputs
st.sidebar.title("Data & Model Controls")
true_a = st.sidebar.number_input('True slope (a)', value=2.0, step=0.1, format="%f")
true_b = st.sidebar.number_input('True intercept (b)', value=1.0, step=0.1, format="%f")
noise = st.sidebar.slider('Noise standard deviation', 0.0, 10.0, 1.0, 0.1)
num_points = st.sidebar.slider('Number of points', 5, 500, 100, 1)
random_seed = st.sidebar.number_input('Random seed (optional)', value=42, step=1)

# Polynomial degree animation controls (global)
st.sidebar.markdown("---")
st.sidebar.header("Animation Controls")
min_deg = st.sidebar.number_input('Min degree', value=1, min_value=1, max_value=10, step=1)
max_deg = st.sidebar.number_input('Max degree', value=5, min_value=1, max_value=20, step=1)
speed = st.sidebar.slider('Animation speed (frames/sec)', 0.5, 5.0, 1.0, 0.1)
st.sidebar.checkbox('Loop animation', value=False, key='loop')
if st.sidebar.button('Play/Pause'):
    st.session_state.play = not st.session_state.play
    st.session_state.paused = False
if st.sidebar.button('Pause/Resume'):
    st.session_state.paused = not st.session_state.paused

# Main layout
st.title("Linear Regression — CRISP-DM Classroom Demo")

# CRISP-DM animated timeline
with st.container():
    st.subheader("CRISP-DM Guided Progress")
    timeline_cols = st.columns([1,1,1,1,1,3])
    steps = ['Business', 'Data', 'Prep', 'Modeling', 'Evaluation']
    # A simple horizontal timeline with animated progress
    prog_placeholder = st.empty()

# Generate data
df = generate_data(true_a, true_b, num_points, noise, random_seed=int(random_seed))

# Tabs
tabs = st.tabs(["Explore", "Compare Models", "Model Insights", "Download Report"])

# ---------- Explore Tab ----------
with tabs[0]:
    st.header("Explore: Build & Visualize a Simple Regression")
    col1, col2 = st.columns([2,1])
    with col1:
        degree = st.slider('Polynomial degree (interactive)', min_value=1, max_value=10, value=1)
        model, y_pred = fit_model(degree, df)
        mse, r2 = compute_metrics(df['y'], y_pred)

        fig = graphobjects.Figure()
        fig.add_trace(graphobjects.Scatter(x=df['x'], y=df['y'], mode='markers', name='Data'))
        # Sort for lines
        xs = np.linspace(df['x'].min(), df['x'].max(), 200)
        if degree == 1:
            coef = model.coef_[0]
            intercept = model.intercept_
            ys = coef * xs + intercept
        else:
            # predict using pipeline
            ys = model.predict(xs.reshape(-1,1))
        fig.add_trace(graphobjects.Scatter(x=xs, y=ys, mode='lines', name=f'Degree {degree} fit'))
        fig.update_layout(title=f'Data and fitted curve (degree {degree}) — MSE={mse:.3f}, R2={r2:.3f}', height=500)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("**Model Metrics**")
        st.write({"MSE": mse, "R2": r2})
        st.markdown("**CRISP-DM Prompt**")
        prompt_text = f"Generate a dataset with a={true_a}, b={true_b}, noise={noise}, n={num_points}. Fit polynomial degree {degree}."
        st.text_area('Prompt (for report)', value=prompt_text, height=120)

# ---------- Compare Models Tab ----------
with tabs[1]:
    st.header("Compare Models: Animated degree sweep")
    comp_col1, comp_col2 = st.columns([3,1])
    with comp_col1:
        comp_deg = st.slider('Compare degrees (select current frame degree)', min_value=min_deg, max_value=max_deg, value=min_deg)
        # Place for animate chart
        comp_chart_placeholder = st.empty()
        comp_prog_ph = st.empty()

        # autoplay controls local to this tab
        autoplay = st.checkbox('Autoplay Compare Models', value=False)
        loop = st.checkbox('Loop', value=False, key='compare_loop')
        play_speed = st.slider('Play speed (frames/sec) - Compare Models', 0.2, 5.0, 1.0, 0.1)
        if st.button('Start Auto-Play (Compare Models)'):
            st.session_state.compare_play = True
            st.session_state.compare_frame = min_deg
        if st.button('Stop Auto-Play'):
            st.session_state.compare_play = False

        # Implement simple autoplay by iterating degrees
        if 'compare_play' not in st.session_state:
            st.session_state.compare_play = False
            st.session_state.compare_frame = min_deg

        if st.session_state.compare_play or autoplay:
            # run a single step then rerun to simulate animation (Streamlit runs top to bottom)
            cur = st.session_state.compare_frame
            # draw chart for cur
            model_cur, y_pred_cur = fit_model(cur, df)
            mse_cur, r2_cur = compute_metrics(df['y'], y_pred_cur)
            figc = graphobjects.Figure()
            figc.add_trace(graphobjects.Scatter(x=df['x'], y=df['y'], mode='markers', name='Data'))
            xs = np.linspace(df['x'].min(), df['x'].max(), 200)
            ys = model_cur.predict(xs.reshape(-1,1))
            figc.add_trace(graphobjects.Scatter(x=xs, y=ys, mode='lines', name=f'Degree {cur}'))
            figc.update_layout(title=f'Degree {cur} — MSE={mse_cur:.3f}, R2={r2_cur:.3f}', height=480)
            comp_chart_placeholder.plotly_chart(figc, use_container_width=True)

            # progress bar below chart
            progress_pct = int( (cur - min_deg) / max(1, (max_deg - min_deg)) * 100 )
            comp_prog_ph.progress(progress_pct)

            # advance frame
            time.sleep(1.0 / max(0.1, play_speed))
            next_frame = cur + 1
            if next_frame > max_deg:
                if loop:
                    next_frame = min_deg
                else:
                    st.session_state.compare_play = False
                    next_frame = cur
            st.session_state.compare_frame = next_frame
            st.experimental_rerun()
        else:
            # static display
            model_cur, y_pred_cur = fit_model(comp_deg, df)
            mse_cur, r2_cur = compute_metrics(df['y'], y_pred_cur)
            figc = graphobjects.Figure()
            figc.add_trace(graphobjects.Scatter(x=df['x'], y=df['y'], mode='markers', name='Data'))
            xs = np.linspace(df['x'].min(), df['x'].max(), 200)
            ys = model_cur.predict(xs.reshape(-1,1))
            figc.add_trace(graphobjects.Scatter(x=xs, y=ys, mode='lines', name=f'Degree {comp_deg}'))
            figc.update_layout(title=f'Degree {comp_deg} — MSE={mse_cur:.3f}, R2={r2_cur:.3f}', height=480)
            comp_chart_placeholder.plotly_chart(figc, use_container_width=True)
            comp_prog_ph.progress(int((comp_deg-min_deg)/max(1,(max_deg-min_deg))*100))

    with comp_col2:
        st.markdown("**Comparison Metrics Table**")
        degrees = list(range(min_deg, max_deg+1))
        rows = []
        for d in degrees:
            _, yp = fit_model(d, df)
            mse_d, r2_d = compute_metrics(df['y'], yp)
            rows.append({"degree": d, "mse": mse_d, "r2": r2_d})
        comp_df = pd.DataFrame(rows)
        st.dataframe(comp_df)

# ---------- Model Insights Tab ----------
with tabs[2]:
    st.header("Model Insights & Residual Animation")
    insight_col1, insight_col2 = st.columns([2,1])
    with insight_col1:
        deg_insight = st.slider('Residuals degree (for residual plot)', 1, 10, 1)
        model_ins, y_pred_ins = fit_model(deg_insight, df)
        residuals = df['y'] - y_pred_ins
        # Animated residuals: We'll animate by coloring points over time (frame index)
        frame_count = 30
        # compute a frame index based on session_state.frame
        frame_idx = int(st.session_state.frame % frame_count)
        # color mapping by frame: highlight a slice
        highlight_idx = int((frame_idx / frame_count) * len(df))
        colors = ['rgba(0,0,0,0.3)'] * len(df)
        for i in range(max(0, highlight_idx-5), min(len(df), highlight_idx+5)):
            colors[i] = 'rgba(255,0,0,0.9)'
        figr = graphobjects.Figure()
        figr.add_trace(graphobjects.Scatter(x=df['x'], y=residuals, mode='markers', marker=dict(color=colors), name='Residuals'))
        figr.add_trace(graphobjects.Scatter(x=[df['x'].min(), df['x'].max()], y=[0,0], mode='lines', name='Zero'))
        figr.update_layout(title=f'Residuals (degree {deg_insight})', height=500)
        st.plotly_chart(figr, use_container_width=True)

    with insight_col2:
        st.markdown("**Residuals statistics**")
        st.write({"mean": float(np.mean(residuals)), "std": float(np.std(residuals))})
        # small control panel for playing the global frame animation
        if st.button('Advance Frame'):
            st.session_state.frame += 1
            st.experimental_rerun()
        if st.button('Reset Frame'):
            st.session_state.frame = 0
            st.experimental_rerun()

# ---------- Download Report Tab ----------
with tabs[3]:
    st.header("Generate & Download PDF Report")
    st.markdown("This report includes the CRISP-DM steps, prompt, data snapshot, and a model summary.")

    # Create CRISP-DM text
    data_description = f"Synthetic dataset with slope={true_a}, intercept={true_b}, noise_std={noise}, n={num_points}."
    modeling_choice = f"Polynomial regression with user-selectable degree(s)."
    evaluation_summary = "MSE and R2 reported; visual residual inspection included."
    crisp_steps = create_crisp_dm_text(data_description, modeling_choice, evaluation_summary)

    progress_percent = int((st.session_state.frame % 100))

    model_info = {
        'selected_degree': degree,
        'mse': f"{mse:.4f}",
        'r2': f"{r2:.4f}"
    }

    if st.button('Create PDF Report'):
        buffer = BytesIO()
        pdf = generate_pdf_report(buffer, crisp_steps, prompt_text, progress_percent, df, model_info)
        st.session_state.generated_pdf = pdf.read()
        st.success('PDF generated — use the Download button below')

    if 'generated_pdf' in st.session_state:
        st.download_button('Download Report PDF', data=st.session_state.generated_pdf, file_name='linear_regression_report.pdf', mime='application/pdf')

# ---------- Background: simple timeline update (visual only) ----------
# We'll update a visual timeline progress bar at top
# A small state machine to advance timeline when play is True
if st.session_state.play:
    # advance a frame based on speed
    now = time.time()
    delta = now - st.session_state.last_update
    frames_to_advance = int(delta * speed)
    if frames_to_advance >= 1:
        st.session_state.frame += frames_to_advance
        st.session_state.last_update = now
        # cap frame to some number for timeline
        st.session_state.frame = st.session_state.frame % 100
        st.experimental_rerun()

# Render timeline (static-ish)
progress_val = int((st.session_state.frame % 100))
prog_cols = st.columns([1,1,1,1,1])
for i, name in enumerate(steps):
    val = min(100, max(0, progress_val - i*20))
    with prog_cols[i]:
        st.metric(name, f"{min(100, max(0, val))}%")

st.markdown("---")
st.caption("Tip: use the sidebar controls to play, pause, loop and control speed. The Compare Models tab contains an auto-play flow with its own progress bar displayed under the chart.")

# ---------- Footer: credits ----------
st.write("Made for classroom demo — includes CRISP-DM steps and interactive controls. PDF generation uses reportlab.")

# End of app
