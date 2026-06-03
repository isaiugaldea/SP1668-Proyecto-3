import os
import joblib
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
import streamlit as st

@st.cache_resource
def load_ml_artifacts(project_dir):
    """
    Carga de forma segura el modelo y todos los preprocesadores (scaler, encoders y columnas).
    Utiliza cache_resource de Streamlit para cargarlos una única vez en memoria.
    """
    artifacts = {}
    
    # Lista de archivos requeridos y sus nombres clave correspondientes
    files = {
        "model": "best_xgb_model.joblib",
        "scaler": "scaler.joblib",
        "label_encoder": "label_encoder_income.joblib",
        "one_hot_encoder": "one_hot_encoder.joblib",
        "columns": "columnas_modelo.joblib"
    }
    
    for key, filename in files.items():
        filepath = os.path.join(project_dir, filename)
        if os.path.exists(filepath):
            try:
                artifacts[key] = joblib.load(filepath)
            except Exception as e:
                st.error(f"Error al cargar {filename}: {e}")
                artifacts[key] = None
        else:
            st.error(f"Archivo no encontrado: {filepath}")
            artifacts[key] = None
            
    return artifacts

@st.cache_data
def load_reference_data(project_dir):
    """
    Carga el dataset de referencia (df_limpio.csv).
    Utiliza cache_data ya que los datos de referencia son estáticos y de lectura.
    """
    filepath = os.path.join(project_dir, "df_limpio.csv")
    if os.path.exists(filepath):
        try:
            return pd.read_csv(filepath)
        except Exception as e:
            st.error(f"Error al cargar df_limpio.csv: {e}")
            return None
    else:
        st.error(f"Archivo no encontrado: {filepath}")
        return None

def preprocess_data(df_input, artifacts):
    """
    Preprocesa un DataFrame de entrada (sea de 1 fila o de N filas) aplicando:
    1. Codificación Ordinal (LabelEncoder) para income_level.
    2. Codificación One-Hot (OneHotEncoder) para las categóricas nominales.
    3. Reordenamiento e integración en el formato exacto de 'columnas_modelo.joblib'.
    4. Escalado final con el 'scaler.joblib'.
    """
    # 1. Obtener objetos de preprocesamiento y columnas esperadas
    scaler = artifacts["scaler"]
    label_encoder = artifacts["label_encoder"]
    one_hot_encoder = artifacts["one_hot_encoder"]
    model_columns = artifacts["columns"]
    
    df_temp = df_input.copy()
    
    # 2. Preprocesamiento de variables individuales:
    # A. Label Encoding de income_level
    if "income_level" in df_temp.columns:
        # Asegurarse de que el valor sea de tipo str y mapear con el label encoder
        df_temp["income_level"] = label_encoder.transform(df_temp["income_level"].astype(str))
        
    # B. One-Hot Encoding de nominales
    # Del comando de inspección sabemos que OHE se ajustó sobre:
    # ['gender', 'ethnicity', 'family_history_diabetes', 'hypertension_history', 'cardiovascular_history', 'smoking_status']
    nominal_cols = ['gender', 'ethnicity', 'family_history_diabetes', 'hypertension_history', 'cardiovascular_history', 'smoking_status']
    
    # Comprobar que todas las columnas nominales existan en la entrada
    for col in nominal_cols:
        if col not in df_temp.columns:
            # Rellenar con un valor por defecto si falta
            if col in ['family_history_diabetes', 'hypertension_history', 'cardiovascular_history']:
                df_temp[col] = 0
            elif col == 'gender':
                df_temp[col] = 'Female'
            elif col == 'ethnicity':
                df_temp[col] = 'White'
            else:
                df_temp[col] = 'Never'

    # Aplicar One-Hot Encoder
    ohe_sparse = one_hot_encoder.transform(df_temp[nominal_cols])
    # Convertir a array denso si es matriz dispersa
    ohe_dense = ohe_sparse.toarray() if hasattr(ohe_sparse, "toarray") else ohe_sparse
    
    # Obtener los nombres de las columnas resultantes del one-hot
    ohe_feature_names = one_hot_encoder.get_feature_names_out(nominal_cols)
    df_ohe = pd.DataFrame(ohe_dense, columns=ohe_feature_names, index=df_temp.index)
    
    # C. Columnas numéricas a conservar directamente antes del escalado
    numeric_cols = [
        'age', 'alcohol_consumption_per_week', 'physical_activity_minutes_per_week', 
        'sleep_hours_per_day', 'screen_time_hours_per_day', 'bmi', 'heart_rate', 
        'cholesterol_total', 'triglycerides', 'insulin_level'
    ]
    
    # Unir numéricas, income_level y las nominales codificadas
    df_combined = pd.concat([df_temp[numeric_cols + ["income_level"]], df_ohe], axis=1)
    
    # D. Asegurar que las columnas estén en el orden exacto que espera el escalador/modelo
    # Si alguna columna falta por alguna razón, la creamos con ceros
    for col in model_columns:
        if col not in df_combined.columns:
            df_combined[col] = 0.0
            
    df_final = df_combined[model_columns]
    
    # 3. Aplicar StandardScaler
    scaled_data = scaler.transform(df_final)
    
    # Devolver como DataFrame estructurado con nombres de columnas originales del modelo
    return pd.DataFrame(scaled_data, columns=model_columns, index=df_input.index)

def calculate_psi(reference, current, num_bins=10):
    """
    Calcula el Population Stability Index (PSI) entre dos distribuciones numéricas.
    
    PSI = sum((Actual_i - Reference_i) * ln(Actual_i / Reference_i))
    """
    reference = np.array(reference)
    current = np.array(current)
    
    # Si ambas poblaciones tienen el mismo valor constante, el PSI es 0
    if np.all(reference == reference[0]) and np.all(current == current[0]) and reference[0] == current[0]:
        return 0.0
        
    # Definir los cuantiles basados en la población de referencia para crear los bins
    percentiles = np.linspace(0, 100, num_bins + 1)
    # Evitar cuantiles duplicados en distribuciones con poca varianza usando percentiles únicos
    bins = np.percentile(reference, percentiles)
    bins = np.unique(bins)
    
    # Si los bins no son suficientes para dividir los datos, usar bins de igual ancho
    if len(bins) < 2:
        bins = np.linspace(min(reference.min(), current.min()) - 1e-5, 
                           max(reference.max(), current.max()) + 1e-5, 
                           num_bins + 1)
    else:
        # Asegurar que cubra los extremos de ambas muestras
        bins[0] = min(bins[0], reference.min(), current.min()) - 1e-5
        bins[-1] = max(bins[-1], reference.max(), current.max()) + 1e-5
        
    # Calcular frecuencias en los intervalos para ambas poblaciones
    ref_counts, _ = np.histogram(reference, bins=bins)
    curr_counts, _ = np.histogram(current, bins=bins)
    
    # Convertir a porcentajes (frecuencia relativa)
    ref_probs = ref_counts / len(reference)
    curr_probs = curr_counts / len(current)
    
    # Manejar frecuencias cero sumando un pequeño epsilon para evitar divisiones entre cero y log(0)
    eps = 1e-4
    ref_probs = np.where(ref_probs == 0, eps, ref_probs)
    curr_probs = np.where(curr_probs == 0, eps, curr_probs)
    
    # Normalizar de nuevo para sumar 1 tras la adición del epsilon
    ref_probs /= np.sum(ref_probs)
    curr_probs /= np.sum(curr_probs)
    
    # Fórmula del PSI
    psi_value = np.sum((curr_probs - ref_probs) * np.log(curr_probs / ref_probs))
    return float(psi_value)

def analyze_dataset_drift(reference_df, current_df, numeric_cols):
    """
    Analiza el data drift para un conjunto de variables numéricas.
    Calcula:
    1. PSI (Population Stability Index)
    2. Estadístico KS y p-valor de Kolmogorov-Smirnov
    
    Devuelve un diccionario con los resultados detallados y el estado del drift.
    """
    drift_report = {}
    
    for col in numeric_cols:
        ref_data = reference_df[col].dropna()
        curr_data = current_df[col].dropna()
        
        # 1. Calcular PSI
        psi_val = calculate_psi(ref_data, curr_data)
        
        # 2. Realizar test de Kolmogorov-Smirnov
        ks_stat, p_val = ks_2samp(ref_data, curr_data)
        
        # Clasificación del Drift
        # Un PSI >= 0.25 indica drift severo, 0.10 <= PSI < 0.25 es moderado, y < 0.10 es estable.
        # Adicionalmente, consideramos si el p-valor de KS es menor a 0.05 (estadísticamente significativo).
        if psi_val >= 0.25:
            severity = "Severe"
            color = "red"
            msg = "Drift Severo Detectado"
        elif psi_val >= 0.10:
            severity = "Warning"
            color = "yellow"
            msg = "Drift Moderado Detectado"
        else:
            severity = "Stable"
            color = "green"
            msg = "Estable"
            
        drift_report[col] = {
            "psi": psi_val,
            "ks_stat": ks_stat,
            "p_value": p_val,
            "severity": severity,
            "color": color,
            "message": msg,
            "stat_significant": p_val < 0.05
        }
        
    return drift_report
