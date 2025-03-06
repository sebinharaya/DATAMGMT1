import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

### 🔹 **1. Configuración de la Aplicación Streamlit** ###
st.set_page_config(page_title="Data Pipeline & ML App", layout="wide")

st.title("📊 Data Pipeline & Machine Learning App")
st.write("Visualización del procesamiento de datos y resultados del modelo de IA.")

### 🔹 **2. Agregar Barra Lateral con Información Personalizada** ###
st.sidebar.markdown("## 🎓 **Lead University**")
st.sidebar.markdown("### 📚 Curso de Administración de Datos")
st.sidebar.markdown("#### 🏆 Trabajo #1: Construcción completa de un Data Pipeline y aplicación de un Modelo de IA")
st.sidebar.markdown("✍️ **Elaborado por:** *Sebastián Ledezma*")
st.sidebar.markdown("📅 **Primer Cuatrimestre de 2025**")

st.sidebar.markdown("---")  # Línea divisoria estilizada

### 🔹 **3. Elegir Paleta de Colores** ###
palette_options = {
    "Tonos Pastel Azul": ["#AED6F1", "#85C1E9", "#5DADE2", "#3498DB", "#2E86C1"],
    "Tonos Pastel Verde": ["#A9DFBF", "#7DCEA0", "#52BE80", "#27AE60", "#1E8449"],
    "Tonos Pastel Morado": ["#D2B4DE", "#AF7AC5", "#9B59B6", "#7D3C98", "#5B2C6F"]
}

chosen_palette = st.sidebar.selectbox("🎨 **Elige una paleta de colores:**", list(palette_options.keys()))

### 🔹 **4. Cargar Datos Procesados** ###
file_path = r"C:\Users\ledez\Desktop\datos_limpios.csv"  # Ruta del CSV
df = pd.read_csv(file_path)

st.subheader("📂 Dataset Procesado")
st.write("Vista previa de los datos:")
st.dataframe(df.head())

### 🔹 **5. Preparar Datos para el Modelo de IA** ###
# Convertir variables categóricas en numéricas (one-hot encoding)
categorical_columns = df.select_dtypes(include=["object"]).columns.tolist()
df_encoded = pd.get_dummies(df, columns=categorical_columns, drop_first=True)

# Definir variables predictoras y objetivo
features = [col for col in df_encoded.columns if col not in ["G1", "G2", "G3", "G-Average", "decrypted_id"]]
X = df_encoded[features]
y = df_encoded["G-Average"]

# Asegurar que todas las variables en X son numéricas
X = X.apply(pd.to_numeric, errors="coerce")

# Dividir los datos en entrenamiento (80%) y prueba (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

### 🔹 **6. Entrenar el Modelo Random Forest** ###
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

### 🔹 **7. Evaluación del Modelo** ###
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

st.subheader("📊 Evaluación del Modelo")
st.write(f"🔹 **MAE (Error Absoluto Medio):** {mae:.2f}")
st.write(f"🔹 **MSE (Error Cuadrático Medio):** {mse:.2f}")
st.write(f"🔹 **RMSE (Raíz del Error Cuadrático Medio):** {rmse:.2f}")
st.write(f"🔹 **R² (Coeficiente de Determinación):** {r2:.2f}")

### 🔹 **8. Importancia de Variables con Gráfico Interactivo** ###
importances = model.feature_importances_
feature_importance_df = pd.DataFrame({"Feature": X.columns, "Importance": importances})
feature_importance_df = feature_importance_df.sort_values(by="Importance", ascending=False)

st.subheader("📌 Importancia de las Variables")

# Selector interactivo para mostrar el Top N variables más importantes
top_n = st.sidebar.radio("📊 **Mostrar Top N Variables:**", [10, 7, 5, 3])

# Mostrar tabla con las N variables más importantes
st.dataframe(feature_importance_df.head(top_n))

# Graficar importancia de variables con la paleta seleccionada
fig, ax = plt.subplots(figsize=(7, 4))  # Ajustar tamaño del gráfico
colors = palette_options[chosen_palette][:top_n]  # Elegir colores de la paleta

sns.barplot(y=feature_importance_df["Feature"][:top_n],
            x=feature_importance_df["Importance"][:top_n],
            palette=colors,
            ax=ax)

ax.set_xlabel("Importancia", fontsize=12)
ax.set_ylabel("Variable", fontsize=12)
ax.set_title("🔹 Importancia de las Variables en el Modelo", fontsize=14)
ax.tick_params(axis='both', labelsize=10)  # Ajustar tamaño de texto

st.pyplot(fig)

