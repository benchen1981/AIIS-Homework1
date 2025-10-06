# 5. ReplitDeploy.md – Step-by-Step Deployment Instructions for Replit

### 1. Setup Project in Replit
1. Go to [https://replit.com](https://replit.com) and create a new Python project.
2. Upload the following files:
   - `streamlit_linear_regression_app.py`
   - `requirements.txt` (if created)
   - Documentation files (optional)

### 2. Add Dependencies
In the Replit shell, run:
```bash
pip install streamlit numpy pandas scikit-learn plotly reportlab
```

Alternatively, add the above lines to a `requirements.txt` file so Replit installs them automatically.

### 3. Configure Replit to Run Streamlit
Edit the **.replit** file and set:
```ini
run = "streamlit run streamlit_linear_regression_app.py --server.port=3000 --server.address=0.0.0.0"
```

### 4. Start the App
Click the **Run** button in Replit. After a few seconds, a Streamlit interface will appear in the web preview panel.

### 5. Using the App in Replit
- Adjust parameters from the sidebar.
- Explore the CRISP-DM timeline and animations.
- Generate and download PDF reports.

### 6. Sharing Your App
- Once running, click the **Share** button in Replit to get a public link.
- Send this link to students, colleagues, or include it in your portfolio.

### 7. Optional: Keep App Alive
For persistent demos, use Replit’s “Always On” feature (available in paid plans) or integrate with UptimeRobot to ping your app periodically.
