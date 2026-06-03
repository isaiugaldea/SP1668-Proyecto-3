import warnings
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import os
import utils

warnings.filterwarnings("ignore")

# Configuración de la página de Streamlit
st.set_page_config(
    page_title="MLOps Dashboard - Model Production & Data Drift",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Estilos CSS Premium (Modo Oscuro, Glassmorphism, Bordes suaves, Semáforo con animaciones)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    /* Fondo oscuro premium */
    .stApp {
        background: linear-gradient(135deg, #0d121d 0%, #151d2a 100%);
        color: #f0f3f8;
    }
    
    /* Contenedor tipo Tarjeta Glassmorphic */
    .glass-card {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 24px;
        backdrop-filter: blur(10px);
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
        margin-bottom: 20px;
    }
    
    /* Tarjeta para Predicción Positiva (Diabetes) */
    .prediction-positive {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.15) 0%, rgba(220, 38, 38, 0.05) 100%);
        border: 1px solid rgba(239, 68, 68, 0.3);
        border-radius: 16px;
        padding: 24px;
        backdrop-filter: blur(10px);
        box-shadow: 0 8px 32px 0 rgba(239, 68, 68, 0.1);
        text-align: center;
        margin-bottom: 20px;
    }
    
    /* Tarjeta para Predicción Negativa (Sano) */
    .prediction-negative {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.15) 0%, rgba(5, 150, 105, 0.05) 100%);
        border: 1px solid rgba(16, 185, 129, 0.3);
        border-radius: 16px;
        padding: 24px;
        backdrop-filter: blur(10px);
        box-shadow: 0 8px 32px 0 rgba(16, 185, 129, 0.1);
        text-align: center;
        margin-bottom: 20px;
    }
    
    /* Animaciones y estilos para los semáforos */
    .status-banner {
        display: flex;
        align-items: center;
        padding: 20px;
        border-radius: 16px;
        margin-bottom: 25px;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.15);
    }
    
    .status-indicator {
        width: 25px;
        height: 25px;
        border-radius: 50%;
        margin-right: 20px;
        flex-shrink: 0;
    }
    
    .status-content h3 {
        margin: 0;
        font-size: 1.4rem;
        font-weight: 700;
    }
    
    .status-content p {
        margin: 5px 0 0 0;
        font-size: 0.95rem;
        opacity: 0.85;
    }
    
    /* Estilos particulares para cada estado del semáforo */
    .stable-glow {
        background: linear-gradient(90deg, rgba(16, 185, 129, 0.2) 0%, rgba(16, 185, 129, 0.05) 100%);
        border: 1px solid rgba(16, 185, 129, 0.4);
    }
    .stable-glow .status-indicator {
        box-shadow: 0 0 15px #10b981;
        animation: pulse-green 2s infinite;
    }
    
    .warning-pulse {
        background: linear-gradient(90deg, rgba(245, 158, 11, 0.2) 0%, rgba(245, 158, 11, 0.05) 100%);
        border: 1px solid rgba(245, 158, 11, 0.4);
    }
    .warning-pulse .status-indicator {
        box-shadow: 0 0 15px #f59e0b;
        animation: pulse-yellow 2s infinite;
    }
    
    .danger-pulse {
        background: linear-gradient(90deg, rgba(239, 68, 68, 0.25) 0%, rgba(239, 68, 68, 0.05) 100%);
        border: 1px solid rgba(239, 68, 68, 0.4);
    }
    .danger-pulse .status-indicator {
        box-shadow: 0 0 15px #ef4444;
        animation: pulse-red 1.5s infinite;
    }
    
    @keyframes pulse-green {
        0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7); }
        70% { transform: scale(1); box-shadow: 0 0 0 12px rgba(16, 185, 129, 0); }
        100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0); }
    }
    @keyframes pulse-yellow {
        0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(245, 158, 11, 0.7); }
        70% { transform: scale(1); box-shadow: 0 0 0 12px rgba(245, 158, 11, 0); }
        100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(245, 158, 11, 0); }
    }
    @keyframes pulse-red {
        0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(239, 68, 68, 0.8); }
        70% { transform: scale(1); box-shadow: 0 0 0 12px rgba(239, 68, 68, 0); }
        100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(239, 68, 68, 0); }
    }
    
    /* Personalización de los Headers */
    .main-title {
        background: linear-gradient(45deg, #3b82f6, #8b5cf6, #ec4899);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        font-size: 2.8rem;
        margin-bottom: 5px;
    }
    
    .subtitle {
        color: #94a3b8;
        font-size: 1.1rem;
        margin-bottom: 30px;
    }

    /* Floating status widget */
    .floating-status {
        position: fixed;
        top: 60px;
        right: 25px;
        z-index: 9999;
        background: rgba(13, 18, 29, 0.85);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 10px 16px;
        backdrop-filter: blur(12px);
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.4);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        max-width: 140px;
        max-height: 42px;
        overflow: hidden;
        white-space: nowrap;
        cursor: pointer;
    }
    
    .floating-status:hover {
        max-width: 320px;
        max-height: 500px;
        background: rgba(21, 29, 42, 0.98);
        border: 1px solid rgba(59, 130, 246, 0.4);
        box-shadow: 0 8px 32px 0 rgba(59, 130, 246, 0.15);
    }
    
    .floating-status-header {
        display: flex;
        align-items: center;
        font-weight: 600;
        font-size: 0.9rem;
        color: #3b82f6;
    }
    
    .floating-status-header span {
        margin-right: 8px;
        font-size: 1.1rem;
    }
    
    .floating-status-content {
        margin-top: 10px;
        opacity: 0;
        visibility: hidden;
        transition: opacity 0.2s ease-in-out, visibility 0.2s;
        font-size: 0.85rem;
        color: #94a3b8;
    }
    .floating-status:hover .floating-status-content {
        opacity: 1;
        visibility: visible;
        white-space: normal;
    }
    
    .status-dot.ok {
        background-color: #10b981;
        box-shadow: 0 0 8px #10b981;
    }
    
    .status-dot.err {
        background-color: #ef4444;
        box-shadow: 0 0 8px #ef4444;
    }
    
    /* Forzar que las etiquetas de todos los widgets sean de color blanco y legibles */
    .stApp label,
    .stApp .stWidgetLabel,
    .stApp [data-testid="stWidgetLabel"] p,
    .stApp [data-testid="stWidgetLabel"],
    .stSlider label,
    .stSelectbox label {
        color: #ffffff !important;
        font-weight: 500 !important;
    }
</style>
""", unsafe_allow_html=True)

# ----------------- CARGA DE DATOS Y ARTEFACTOS -----------------
project_dir = os.path.dirname(os.path.abspath(__file__))

artifacts = utils.load_ml_artifacts(project_dir)
df_ref = utils.load_reference_data(project_dir)

success_load = True
status_items_html = ""

for key, filename in {
    "Modelo (XGBoost)": "best_xgb_model.joblib",
    "Escalador (StandardScaler)": "scaler.joblib",
    "Label Encoder (Income)": "label_encoder_income.joblib",
    "One-Hot Encoder": "one_hot_encoder.joblib",
    "Lista de Columnas": "columnas_modelo.joblib",
    "Dataset Referencia (CSV)": "df_limpio.csv"
}.items():
    check_key = "model" if "Modelo" in key else ("scaler" if "Escalador" in key else ("label_encoder" if "Label" in key else ("one_hot_encoder" if "One-Hot" in key else ("columns" if "Lista" in key else "csv"))))
    
    is_ok = False
    if check_key == "csv":
        is_ok = df_ref is not None
    else:
        is_ok = artifacts.get(check_key) is not None
        
    dot_class = "ok" if is_ok else "err"
    status_label = "Cargado" if is_ok else "Error"
    
    status_items_html += f'<div class="status-item"><span class="status-dot {dot_class}"></span><span><strong>{key}</strong>: {status_label}</span></div>'
    if not is_ok:
        success_load = False

# Renderizar el widget flotante premium en HTML (sin saltos de línea con sangría para evitar interpretación como código markdown)
floating_status_html = f'<div class="floating-status"><div class="floating-status-header"><span>🛠️</span> MLOps Status</div><div class="floating-status-content"><hr style="margin: 8px 0; border-color: rgba(255,255,255,0.1);">{status_items_html}</div></div>'
st.markdown(floating_status_html, unsafe_allow_html=True)

if not success_load:
    st.error("Error crítico: No se pudieron cargar todos los artefactos de ML necesarios. Por favor verifica que los archivos .joblib y .csv estén en el directorio correcto.")
    st.stop()

# Listado de variables numéricas y categóricas del dataset
numeric_cols = [
    'age', 'alcohol_consumption_per_week', 'physical_activity_minutes_per_week', 
    'sleep_hours_per_day', 'screen_time_hours_per_day', 'bmi', 'heart_rate', 
    'cholesterol_total', 'triglycerides', 'insulin_level'
]
categorical_cols = ['gender', 'ethnicity', 'income_level', 'smoking_status']
binary_cols = ['family_history_diabetes', 'hypertension_history', 'cardiovascular_history']

# Header Principal de la Aplicación
st.markdown("<div class='main-title'>Model Sandbox & Data Drift Monitor</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Interfaz de Producción y Simulación de Estrés para Modelos Predictivos de Salud</div>", unsafe_allow_html=True)

# Creación de Pestañas (Tabs)
tab1, tab2 = st.tabs(["🎯 Simulador de Predicciones Individuales", "⚡ Laboratorio de Data Drift & Estrés"])

# ==========================================
# PESTAÑA 1: PREDICCIONES INDIVIDUALES
# ==========================================
with tab1:
    st.markdown("### Ajuste de Características Individuales")
    st.write("Modifica los controles a continuación para ingresar los datos de un paciente. La predicción del modelo se actualizará instantáneamente.")
    
    # 1. Crear contenedores para controlar el orden vertical en la interfaz
    banner_container = st.container()
    charts_container = st.container()
    
    st.markdown("---")
    
    # 2. Contenedor para parámetros de entrada (en la parte inferior)
    inputs_container = st.container()
    
    with inputs_container:
        st.markdown("#### 🛠️ Ajuste de Parámetros del Paciente")
        
        # Organizar parámetros en 3 columnas horizontales
        col_in1, col_in2, col_in3 = st.columns(3, gap="medium")
        
        with col_in1:
            st.markdown("<div class='glass-card' style='height: 100%; margin-bottom: 0;'>", unsafe_allow_html=True)
            st.markdown("##### 🩺 Parámetros Fisiológicos")
            age = st.slider("Edad (Años)", int(df_ref['age'].min()), int(df_ref['age'].max()), int(df_ref['age'].mean()))
            bmi = st.slider("Índice de Masa Corporal (BMI)", float(df_ref['bmi'].min()), float(df_ref['bmi'].max()), float(df_ref['bmi'].mean()), step=0.1)
            heart_rate = st.slider("Frecuencia Cardíaca (LPM)", int(df_ref['heart_rate'].min()), int(df_ref['heart_rate'].max()), int(df_ref['heart_rate'].mean()))
            insulin_level = st.slider("Nivel de Insulina (µIU/mL)", float(df_ref['insulin_level'].min()), float(df_ref['insulin_level'].max()), float(df_ref['insulin_level'].mean()), step=0.1)
            st.markdown("</div>", unsafe_allow_html=True)

        with col_in2:
            st.markdown("<div class='glass-card' style='height: 100%; margin-bottom: 0;'>", unsafe_allow_html=True)
            st.markdown("##### 🧪 Hábitos & Estilo de Vida")
            cholesterol_total = st.slider("Colesterol Total (mg/dL)", int(df_ref['cholesterol_total'].min()), int(df_ref['cholesterol_total'].max()), int(df_ref['cholesterol_total'].mean()))
            triglycerides = st.slider("Triglicéridos (mg/dL)", int(df_ref['triglycerides'].min()), int(df_ref['triglycerides'].max()), int(df_ref['triglycerides'].mean()))
            alcohol_consumption = st.slider("Consumo Alcohol Semanal (Tragos)", int(df_ref['alcohol_consumption_per_week'].min()), int(df_ref['alcohol_consumption_per_week'].max()), int(df_ref['alcohol_consumption_per_week'].mean()))
            physical_activity = st.slider("Minutos Actividad Física / Sem.", int(df_ref['physical_activity_minutes_per_week'].min()), int(df_ref['physical_activity_minutes_per_week'].max()), int(df_ref['physical_activity_minutes_per_week'].mean()))
            sleep_hours = st.slider("Horas de Sueño / Día", float(df_ref['sleep_hours_per_day'].min()), float(df_ref['sleep_hours_per_day'].max()), float(df_ref['sleep_hours_per_day'].mean()), step=0.5)
            screen_time = st.slider("Tiempo en Pantalla / Día (Horas)", float(df_ref['screen_time_hours_per_day'].min()), float(df_ref['screen_time_hours_per_day'].max()), float(df_ref['screen_time_hours_per_day'].mean()), step=0.5)
            st.markdown("</div>", unsafe_allow_html=True)

        with col_in3:
            st.markdown("<div class='glass-card' style='height: 100%; margin-bottom: 0;'>", unsafe_allow_html=True)
            st.markdown("##### 👥 Historial & Datos Demográficos")
            gender = st.selectbox("Género", options=list(df_ref['gender'].unique()))
            ethnicity = st.selectbox("Etnicidad", options=list(df_ref['ethnicity'].unique()))
            income_level = st.selectbox("Nivel de Ingresos (Orden Ordinal)", options=list(artifacts['label_encoder'].classes_))
            smoking_status = st.selectbox("Estatus de Fumador", options=list(df_ref['smoking_status'].unique()))
            family_history = st.selectbox("Historial Familiar de Diabetes", options=["No", "Sí"])
            hypertension = st.selectbox("Historial de Hipertensión", options=["No", "Sí"])
            cardiovascular = st.selectbox("Historial Cardiovascular", options=["No", "Sí"])
            st.markdown("</div>", unsafe_allow_html=True)

        # Mapeo de campos binarios
        family_history_val = 1 if family_history == "Sí" else 0
        hypertension_val = 1 if hypertension == "Sí" else 0
        cardiovascular_val = 1 if cardiovascular == "Sí" else 0
        
        # Crear diccionario para enviar a preprocesamiento
        raw_sample = {
            'age': age,
            'gender': gender,
            'ethnicity': ethnicity,
            'income_level': income_level,
            'smoking_status': smoking_status,
            'alcohol_consumption_per_week': alcohol_consumption,
            'physical_activity_minutes_per_week': physical_activity,
            'sleep_hours_per_day': sleep_hours,
            'screen_time_hours_per_day': screen_time,
            'family_history_diabetes': family_history_val,
            'hypertension_history': hypertension_val,
            'cardiovascular_history': cardiovascular_val,
            'bmi': bmi,
            'heart_rate': heart_rate,
            'cholesterol_total': cholesterol_total,
            'triglycerides': triglycerides,
            'insulin_level': insulin_level
        }
        
    # 3. Realizar Inferencia
    df_sample = pd.DataFrame([raw_sample])
    df_sample_scaled = utils.preprocess_data(df_sample, artifacts)
    
    model = artifacts["model"]
    pred_class = model.predict(df_sample_scaled)[0]
    pred_proba = model.predict_proba(df_sample_scaled)[0][1] # Probabilidad de clase 1 (Diabetes)
    
    # 4. Renderizar Banner de Resultado en el contenedor superior
    with banner_container:
        if pred_class == 1:
            st.markdown(f"""
            <div class='prediction-positive'>
                <h4 style='color: #ef4444; margin: 0; font-weight: 700;'>RIESGO DETECTADO</h4>
                <h2 style='color: #ef4444; font-size: 2.2rem; margin: 10px 0;'>Diagnóstico: Diabetes (1)</h2>
                <p style='color: #fca5a5; font-size: 0.95rem; margin: 0;'>El modelo estima alta correlación con marcadores clínicos de diabetes. Se recomienda evaluación médica formal.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class='prediction-negative'>
                <h4 style='color: #10b981; margin: 0; font-weight: 700;'>RIESGO BAJO</h4>
                <h2 style='color: #10b981; font-size: 2.2rem; margin: 10px 0;'>Diagnóstico: Normal (0)</h2>
                <p style='color: #a7f3d0; font-size: 0.95rem; margin: 0;'>Los parámetros del paciente están dentro de la distribución estable saludable del conjunto de entrenamiento.</p>
            </div>
            """, unsafe_allow_html=True)
            
    # 5. Renderizar los gráficos lado a lado en el contenedor de gráficos
    with charts_container:
        col_g1, col_g2 = st.columns(2, gap="large")
        
        with col_g1:
            st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
            st.markdown("##### 📈 Probabilidad de Diabetes", unsafe_allow_html=True)
            
            bar_color = "#ef4444" if pred_proba >= 0.70 else ("#f59e0b" if pred_proba >= 0.35 else "#10b981")
            
            fig_gauge = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = pred_proba * 100,
                domain = {'x': [0, 1], 'y': [0, 1]},
                number = {'suffix': "%", 'font': {'size': 44, 'color': '#ffffff', 'family': 'Outfit'}},
                gauge = {
                    'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#4E5D6C", 'tickfont': {'color': '#94a3b8'}},
                    'bar': {'color': bar_color, 'thickness': 0.3},
                    'bgcolor': "rgba(255,255,255,0.03)",
                    'borderwidth': 1,
                    'bordercolor': "rgba(255,255,255,0.1)",
                    'steps': [
                        {'range': [0, 35], 'color': 'rgba(16, 185, 129, 0.1)'},
                        {'range': [35, 70], 'color': 'rgba(245, 158, 11, 0.1)'},
                        {'range': [70, 100], 'color': 'rgba(239, 68, 68, 0.1)'}
                    ],
                    'threshold': {
                        'line': {'color': "#ffffff", 'width': 2},
                        'thickness': 0.8,
                        'value': pred_proba * 100
                    }
                }
            ))
            
            fig_gauge.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font={'color': "#ffffff", 'family': "Outfit"},
                height=220,
                margin=dict(l=20, r=20, t=10, b=10)
            )
            st.plotly_chart(fig_gauge, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
            
        with col_g2:
            st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
            st.markdown("##### 🔍 Comparación con el Promedio de la Población (Z-Score)", unsafe_allow_html=True)
            
            # Obtener medias estandarizadas
            scaler = artifacts["scaler"]
            df_dummy_means = pd.DataFrame(df_ref.mean(numeric_only=True)).T
            for cat_col in categorical_cols:
                df_dummy_means[cat_col] = df_ref[cat_col].mode()[0]
            for bin_col in binary_cols:
                df_dummy_means[bin_col] = 0
                
            df_means_scaled = utils.preprocess_data(df_dummy_means, artifacts)
            
            sample_scaled_vals = df_sample_scaled[numeric_cols].values[0]
            ref_scaled_means = df_means_scaled[numeric_cols].values[0]
            
            fig_compare = go.Figure()
            
            fig_compare.add_trace(go.Bar(
                y=numeric_cols,
                x=sample_scaled_vals,
                orientation='h',
                name='Paciente',
                marker=dict(
                    color='rgba(59, 130, 246, 0.7)',
                    line=dict(color='rgba(59, 130, 246, 1.0)', width=1.5)
                )
            ))
            
            fig_compare.add_trace(go.Scatter(
                y=numeric_cols,
                x=ref_scaled_means,
                mode='markers+lines',
                name='Media de Referencia',
                line=dict(color='#f59e0b', width=2, dash='dash'),
                marker=dict(size=8, color='#f59e0b')
            ))
            
            fig_compare.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font={'color': "#ffffff", 'family': "Outfit"},
                height=220,
                margin=dict(l=10, r=10, t=10, b=10),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                xaxis=dict(gridcolor="rgba(255,255,255,0.05)", title="Z-Score"),
                yaxis=dict(gridcolor="rgba(255,255,255,0.05)")
            )
            st.plotly_chart(fig_compare, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

# ==========================================
# PESTAÑA 2: LABORATORIO E INTERFAZ DE DATA DRIFT
# ==========================================
with tab2:
    st.markdown("### Laboratorio de Simulación de Estrés y Data Drift")
    st.write("Actúa como el 'villano' modificando las condiciones macro de la población. Introduce distorsiones y observa cómo el semáforo de degradación y las pruebas estadísticas PSI y Kolmogorov-Smirnov reaccionan en tiempo real.")
    
    # 1. Crear contenedores para controlar el orden vertical en la interfaz
    drift_banner_container = st.container()
    drift_results_container = st.container()
    
    st.markdown("---")
    
    # 2. Contenedor para inyección de drift (en la parte inferior)
    drift_inputs_container = st.container()
    
    with drift_inputs_container:
        st.markdown("#### 🧬 Inyección de Drift Poblacional")
        st.write("Ajusta los siguientes controles para alterar las distribuciones del conjunto de prueba actual frente al de entrenamiento (Referencia).")
        
        col_dr1, col_dr2, col_dr3 = st.columns(3, gap="medium")
        
        with col_dr1:
            st.markdown("<div class='glass-card' style='height: 100%; margin-bottom: 0;'>", unsafe_allow_html=True)
            drift_age_shift = st.slider("Desplazar Edad Promedio (Años)", -25, 25, 0, key="age_shift_drift")
            drift_bmi_mult = st.slider("Multiplicador Global de BMI (Obesidad)", 0.6, 1.8, 1.0, step=0.05, key="bmi_mult_drift")
            st.markdown("</div>", unsafe_allow_html=True)
            
        with col_dr2:
            st.markdown("<div class='glass-card' style='height: 100%; margin-bottom: 0;'>", unsafe_allow_html=True)
            drift_phys_mult = st.slider("Multiplicador de Actividad Física", 0.2, 1.8, 1.0, step=0.05, key="phys_mult_drift")
            drift_insulin_mult = st.slider("Multiplicador de Niveles de Insulina", 0.5, 2.5, 1.0, step=0.05, key="ins_mult_drift")
            st.markdown("</div>", unsafe_allow_html=True)
            
        with col_dr3:
            st.markdown("<div class='glass-card' style='height: 100%; margin-bottom: 0;'>", unsafe_allow_html=True)
            drift_chol_shift = st.slider("Desplazar Colesterol (mg/dL)", -80, 80, 0, key="chol_shift_drift")
            drift_trig_shift = st.slider("Desplazar Triglicéridos (mg/dL)", -100, 100, 0, key="trig_shift_drift")
            st.markdown("</div>", unsafe_allow_html=True)
            
        # Crear dataset alterado en tiempo real a partir del dataset de referencia
        df_drifted = df_ref.copy()
        
        # Aplicar distorsiones controlando que se mantengan dentro de límites biológicos/lógicos
        if drift_age_shift != 0:
            df_drifted['age'] = (df_drifted['age'] + drift_age_shift).clip(18, 100)
            
        if drift_bmi_mult != 1.0:
            df_drifted['bmi'] = (df_drifted['bmi'] * drift_bmi_mult).clip(10, 70)
            
        if drift_phys_mult != 1.0:
            df_drifted['physical_activity_minutes_per_week'] = (df_drifted['physical_activity_minutes_per_week'] * drift_phys_mult).clip(0, 1500).astype(int)
            
        if drift_insulin_mult != 1.0:
            df_drifted['insulin_level'] = (df_drifted['insulin_level'] * drift_insulin_mult).clip(1, 100)
            
        if drift_chol_shift != 0:
            df_drifted['cholesterol_total'] = (df_drifted['cholesterol_total'] + drift_chol_shift).clip(50, 600)
            
        if drift_trig_shift != 0:
            df_drifted['triglycerides'] = (df_drifted['triglycerides'] + drift_trig_shift).clip(30, 800)

    # 3. Calcular métricas de Drift en tiempo real
    drift_report = utils.analyze_dataset_drift(df_ref, df_drifted, numeric_cols)
    
    # Calcular PSI global promedio
    all_psis = [report["psi"] for report in drift_report.values()]
    avg_psi = np.mean(all_psis)
    max_psi = np.max(all_psis)
    
    # Determinar el estado del semáforo
    if avg_psi >= 0.25 or max_psi >= 0.5:
        semaphore_class = "danger-pulse"
        semaphore_color = "#ef4444"
        semaphore_title = "PELIGRO CRÍTICO - DATA DRIFT SEVERO"
        semaphore_desc = f"El Population Stability Index (PSI Promedio: {avg_psi:.3f}) indica un cambio drástico en las variables. Las predicciones del modelo en producción ya no son estadísticamente confiables. Se requiere reentrenamiento urgente."
    elif avg_psi >= 0.10 or max_psi >= 0.2:
        semaphore_class = "warning-pulse"
        semaphore_color = "#f59e0b"
        semaphore_title = "ADVERTENCIA - DATA DRIFT MODERADO"
        semaphore_desc = f"Se ha detectado un cambio significativo en algunas variables clave (PSI Promedio: {avg_psi:.3f}). Se recomienda monitorear de cerca el rendimiento del modelo en producción."
    else:
        semaphore_class = "stable-glow"
        semaphore_color = "#10b981"
        semaphore_title = "SISTEMA ESTABLE - SIN DRIFT SIGNIFICATIVO"
        semaphore_desc = f"Las distribuciones actuales del sistema coinciden plenamente con el conjunto de entrenamiento (PSI Promedio: {avg_psi:.3f}). El modelo opera bajo condiciones normales."

    # 4. Inyectar el Banner de Semáforo Reactivo en el contenedor de banner
    with drift_banner_container:
        st.markdown(f"""
        <div class="status-banner {semaphore_class}">
            <div class="status-indicator" style="background-color: {semaphore_color};"></div>
            <div class="status-content">
                <h3>{semaphore_title}</h3>
                <p>{semaphore_desc}</p>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # 5. Renderizar el cuadro y el gráfico de distribuciones en el contenedor de resultados
    with drift_results_container:
        col_res_left, col_res_right = st.columns([1.2, 1], gap="large")
        
        with col_res_left:
            # Formatear reporte de drift para visualizar en tabla
            table_data = []
            for col, metrics in drift_report.items():
                if metrics["severity"] == "Severe":
                    status_icon = "🔴 Drift Severo"
                elif metrics["severity"] == "Warning":
                    status_icon = "🟡 Drift Moderado"
                else:
                    status_icon = "🟢 Estable"
                    
                table_data.append({
                    "Variable": col,
                    "PSI (Stability)": round(metrics["psi"], 4),
                    "KS Stat": round(metrics["ks_stat"], 4),
                    "p-valor (KS)": f"{metrics['p_value']:.4e}" if metrics['p_value'] < 0.0001 else round(metrics['p_value'], 5),
                    "Estado del Drift": status_icon,
                    "Diferencia (p < 0.05)": "Sí" if metrics["stat_significant"] else "No"
                })
                
            df_table = pd.DataFrame(table_data)
            
            st.markdown("<div class='glass-card' style='height: 100%;'>", unsafe_allow_html=True)
            st.markdown("##### 📊 Cuadro de Métricas de Drift Poblacional", unsafe_allow_html=True)
            st.write("Estadísticos calculados en tiempo real comparando la población de referencia vs la población modificada.")
            st.dataframe(df_table, use_container_width=True, hide_index=True)
            st.markdown("</div>", unsafe_allow_html=True)
            
        with col_res_right:
            st.markdown("<div class='glass-card' style='height: 100%;'>", unsafe_allow_html=True)
            st.markdown("##### 🔬 Comparación Visual de Distribuciones (KDE / Histogramas)")
            
            # Selector de variable
            selected_col = st.selectbox("Selecciona la variable a analizar:", options=numeric_cols, key="drift_col_select")
            
            # Generar gráfico interactivo de distribución comparando original y actual
            fig_dist = go.Figure()
            
            # Histograma de Referencia
            fig_dist.add_trace(go.Histogram(
                x=df_ref[selected_col],
                name='Referencia (Entrenamiento)',
                nbinsx=30,
                histnorm='probability density',
                marker=dict(
                    color='rgba(16, 185, 129, 0.35)',
                    line=dict(color='rgba(16, 185, 129, 0.8)', width=1.5)
                )
            ))
            
            # Histograma con Drift (Actual)
            fig_dist.add_trace(go.Histogram(
                x=df_drifted[selected_col],
                name='Actual (Simulado)',
                nbinsx=30,
                histnorm='probability density',
                marker=dict(
                    color='rgba(239, 68, 68, 0.35)',
                    line=dict(color='rgba(239, 68, 68, 0.8)', width=1.5)
                )
            ))
            
            fig_dist.update_layout(
                barmode='overlay',
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font={'color': "#ffffff", 'family': "Outfit"},
                height=220,
                margin=dict(l=10, r=10, t=10, b=10),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
                xaxis=dict(gridcolor="rgba(255,255,255,0.05)", title=selected_col),
                yaxis=dict(gridcolor="rgba(255,255,255,0.05)", title="Densidad")
            )
            st.plotly_chart(fig_dist, use_container_width=True)
            
            # Explicar las métricas de la variable seleccionada
            metrics_sel = drift_report[selected_col]
            p_val_formatted = f"{metrics_sel['p_value']:.4e}" if metrics_sel['p_value'] < 0.0001 else f"{metrics_sel['p_value']:.5f}"
            
            if metrics_sel['severity'] == "Stable":
                interp_txt = "Estable (Sin drift significativo)."
                interp_color = "#10b981"
            elif metrics_sel['severity'] == "Warning":
                interp_txt = "Desplazamiento moderado. Vigilar variable."
                interp_color = "#f59e0b"
            else:
                interp_txt = "¡Drift severo! Las predicciones pueden fallar de forma silenciosa."
                interp_color = "#ef4444"
                
            st.markdown(f"""
            <div style="font-size: 0.85rem; padding: 10px; background: rgba(255,255,255,0.02); border-radius: 8px; border: 1px solid rgba(255,255,255,0.05); margin-top: 10px;">
                <strong>PSI:</strong> {metrics_sel['psi']:.4f} | <strong>KS:</strong> {metrics_sel['ks_stat']:.4f} | <strong>p-valor:</strong> {p_val_formatted}<br>
                <strong>Interpretación:</strong> <span style="color: {interp_color};">{interp_txt}</span>
            </div>
            """, unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
