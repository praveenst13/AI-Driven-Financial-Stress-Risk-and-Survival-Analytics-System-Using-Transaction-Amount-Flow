import streamlit as st
import pandas as pd
import numpy as np
import joblib
import time
import plotly.express as px
import plotly.graph_objects as go

# --- 1. SETUP & ASSET LOADING ---
st.set_page_config(page_title="MSME AI Survival Analytics", layout="wide")

@st.cache_resource
def load_assets():
    try:
        scaler = joblib.load('scaler.pkl')
        le = joblib.load('label_encoder.pkl')
        stats = joblib.load('model_stats.pkl')
        models = {
            "XGBoost": joblib.load('XGBoost_model.pkl'),
            "Random Forest": joblib.load('Random_Forest_model.pkl'),
            "Logistic Regression": joblib.load('Logistic_Regression_model.pkl')
        }
        return scaler, le, stats, models
    except Exception as e:
        st.error(f"Error loading pkl files: {e}")
        return None, None, None, None

scaler, le, stats, models = load_assets()

# --- 2. SESSION STATE ---
if 'history' not in st.session_state:
    st.session_state.history = pd.DataFrame(columns=['Time', 'Amount', 'Score', 'Liquidity', 'Balance'])

# --- 3. DASHBOARD HEADER ---
st.title("🛡️ MSME Financial Stress & Survival Analytics")
st.markdown("### Integrated Diagnostic, Predictive, and Prescriptive Intelligence")

# --- 4. TABS FOR ANALYSIS TYPES ---
tab_live, tab_compare, tab_diagnostic = st.tabs([
    "🚀 Live Analytics & Survival", 
    "📊 Model Performance Comparison",
    "🔍 Diagnostic & Prescriptive Insights"
])

# --- TAB 1: LIVE ANALYTICS (Predictive) ---
with tab_live:
    col_ctrl, col_main = st.columns([1, 2])
    
    with col_ctrl:
        st.subheader("Control Panel")
        active_model_name = st.selectbox("Active AI Model", list(models.keys()))
        run_sim = st.toggle("Start Live Transaction Feed")
        stream_speed = st.slider("Update Interval (s)", 1, 5, 2)
        st.divider()
        st.info("**Survival Rule:** If score stays > 70% for 3 cycles, business is flagged for 'Immediate Insolvency Risk'.")

    with col_main:
        st.subheader("Real-Time Stress Meter")
        metric_placeholder = st.empty()
        chart_placeholder = st.empty()

    if run_sim:
        while run_sim:
            # Simulate Data
            new_time = time.strftime("%H:%M:%S")
            amt = np.random.uniform(5000, 150000)
            o_bal = np.random.uniform(20000, 200000)
            n_bal = o_bal - amt
            
            # Feature Engineering (Must match Colab exactly)
            liq = n_bal / (amt + 1)
            dep = (o_bal - n_bal) / (o_bal + 1)
            # 4 = TRANSFER, 0 = High Value Flag
            feat_df = pd.DataFrame([[amt, o_bal, n_bal, liq, dep, 4, 0]], 
                                   columns=['amount', 'oldbalanceOrg', 'newbalanceOrig', 'Liquidity_Ratio', 'Balance_Depletion', 'type_encoded', 'Is_High_Value'])
            
            # Predictive Analysis
            prob = models[active_model_name].predict_proba(scaler.transform(feat_df))[0][1] * 100
            
            # Update History
            new_row = pd.DataFrame({'Time': [new_time], 'Amount': [amt], 'Score': [prob], 'Liquidity': [liq], 'Balance': [n_bal]})
            st.session_state.history = pd.concat([st.session_state.history, new_row]).tail(20)
            
            # Update Dashboard Visuals
            with metric_placeholder.container():
                m1, m2, m3 = st.columns(3)
                m1.metric("Risk Score", f"{prob:.1f}%", delta=f"{prob-20:.1f}%", delta_color="inverse")
                m2.metric("Liquidity Ratio", f"{liq:.2f}")
                m3.metric("Current Balance", f"${n_bal:,.0f}")
            
            with chart_placeholder.container():
                fig = px.area(st.session_state.history, x='Time', y='Score', 
                             title=f"Survival Risk Trend ({active_model_name})",
                             color_discrete_sequence=['#ff4b4b'])
                st.plotly_chart(fig, use_container_width=True)
            
            time.sleep(stream_speed)
            st.rerun()

# --- TAB 2: MODEL COMPARISON (Descriptive/Predictive) ---
with tab_compare:
    st.header("Model Benchmarking")
    if stats:
        df_stats = pd.DataFrame(list(stats.items()), columns=['Model', 'Metric_Text'])
        df_stats['AUC_Value'] = df_stats['Metric_Text'].str.extract(r'(\d+\.\d+)').astype(float)
        
        c1, c2 = st.columns(2)
        with c1:
            st.write("### Training Metrics")
            st.table(df_stats[['Model', 'Metric_Text']])
        with c2:
            fig_comp = px.bar(df_stats.dropna(), x='Model', y='AUC_Value', color='AUC_Value', 
                             title="AI Reliability (AUC Score)", color_continuous_scale='RdYlGn')
            st.plotly_chart(fig_comp, use_container_width=True)
    else:
        st.warning("model_stats.pkl not found.")

# --- TAB 3: DIAGNOSTIC & PRESCRIPTIVE ---
with tab_diagnostic:
    st.header("Deep Analysis & Survival Action Plan")
    
    if not st.session_state.history.empty:
        diag_df = st.session_state.history.copy()
        # FIX: Ensure sizes are positive for Scatter Chart
        diag_df['Bubble_Size'] = diag_df['Liquidity'].abs() + 0.1
        latest_score = diag_df.iloc[-1]['Score']
        
        # 1. Diagnostic Analysis (Why is it happening?)
        st.subheader("Diagnostic: Risk Correlation")
        fig_diag = px.scatter(diag_df, x="Amount", y="Score", size="Bubble_Size", 
                             color="Score", hover_data=['Liquidity', 'Balance'],
                             title="Analysis: Transaction Impact on Financial Stress",
                             color_continuous_scale="RdYlGn_r")
        st.plotly_chart(fig_diag, use_container_width=True)
        
        # 2. Prescriptive Analysis (What should we do?)
        st.divider()
        st.subheader("Prescriptive: Survival Action Plan")
        
        # Survival Runway Calculation
        avg_spend = diag_df['Amount'].mean()
        curr_bal = diag_df.iloc[-1]['Balance']
        runway = curr_bal / avg_spend if avg_spend > 0 else 0
        
        p_col1, p_col2 = st.columns([1, 2])
        p_col1.metric("Estimated Runway", f"{max(0, int(runway))} Transactions")
        
        with p_col2:
            if latest_score > 70:
                st.error("🚨 **CRITICAL SURVIVAL ACTION:**\n- Halt all discretionary spending.\n- Your cash depletion rate is unsustainable.\n- Action: Secure emergency credit or equity injection.")
            elif latest_score > 30:
                st.warning("⚠️ **PRECAUTIONARY STRATEGY:**\n- Optimize accounts receivable to increase inflow.\n- Maintain a liquidity buffer of 20%.\n- Action: Review recurring transfer patterns.")
            else:
                st.success("✅ **STABILITY STRATEGY:**\n- Financial health is optimal.\n- Action: Consider reinvesting surplus into growth or inventory.")
    else:
        st.info("Start the Live Feed in the first tab to generate Diagnostic insights.")