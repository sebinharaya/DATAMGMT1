import pandas as pd

### 1. Selección y Extracción de Datos

### 1.1 Fuente de datos web y extracción automatizada.
url = "https://docs.google.com/spreadsheets/d/e/2PACX-1vT86_UguzsdM8p-Ci1EyLNsfc9vlnIhbvuySKrU-d-EqX3uRA_dJzQCT9LG6ojTBRM5fd3OEZBiXNbA/pub?gid=648965636&single=true&output=csv"
df = pd.read_csv(url, sep=None, engine="python")  # Pandas detecta el delimitador automáticamente
print(df.head())  # Muestra las primeras filas del DataFrame

### 1.2 Seguridad de extracción de datos: encriptación
import gspread
from google.oauth2.service_account import Credentials
import pandas as pd
from cryptography.fernet import Fernet

### 🔐 Cargar o generar clave de encriptación ###
KEY_FILE = "secret.key"

# Generar una clave si no existe
try:
    with open(KEY_FILE, "rb") as key_file:
        key = key_file.read()
except FileNotFoundError:
    key = Fernet.generate_key()
    with open(KEY_FILE, "wb") as key_file:
        key_file.write(key)

cipher = Fernet(key)  # Inicializar cifrador

### 🔑 Conectar con Google Sheets API ###
SHEET_URL = "https://docs.google.com/spreadsheets/d/1yqPLutCt4uDRQCX0fbFkZCVnaH5NJ3Fe5GtUsHbIcNY"

# Cargar credenciales desde archivo JSON
scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
creds = Credentials.from_service_account_file("credentials.json", scopes=scope)
client = gspread.authorize(creds)

# Abrir hoja de cálculo
sheet = client.open_by_url(SHEET_URL)
worksheet = sheet.worksheet("student-por")

# Leer datos como DataFrame
df = pd.DataFrame(worksheet.get_all_records())

### 🔐 Encriptar la columna "personal_id" ###
if "personal_id" in df.columns:
    df['encrypted_id'] = df['personal_id'].apply(lambda x: cipher.encrypt(str(x).encode()).decode())
    df.drop(columns=["personal_id"], inplace=True)  # Eliminar la columna original por seguridad
else:
    print("⚠️ Advertencia: La columna 'personal_id' no existe en la hoja de cálculo.")

### 📊 Mostrar los datos encriptados ###
print(df.head())

### (Opcional) Desencriptar cuando sea necesario ###
def decrypt_id(encrypted_value):
    return cipher.decrypt(encrypted_value.encode()).decode()

# Desencriptar solo cuando lo necesites
df["decrypted_id"] = df["encrypted_id"].apply(decrypt_id)
print(df[["encrypted_id", "decrypted_id"]])

### 1.3 Seguridad de extracción de datos: autenticación
import os
import getpass
from dotenv import load_dotenv

# Cargar variables desde el archivo .env
load_dotenv("admin.env")

USUARIO_CORRECTO = os.getenv("USUARIO_APP")
PASSWORD_CORRECTO = os.getenv("PASSWORD_APP")

# Pedir usuario y contraseña
usuario = input("👤 Usuario: ")
password = getpass.getpass("🔑 Contraseña: ")

# Verificar credenciales
if usuario == USUARIO_CORRECTO and password == PASSWORD_CORRECTO:
    print("✅ Autenticación exitosa. Cargando datos...")
else:
    print("❌ Acceso denegado. Credenciales incorrectas.")
    exit()

### 🧹 **Limpieza y Transformación de Datos** ###
# 1️⃣ **Eliminar duplicados basado en 'decrypted_id' (pero sin eliminar la columna)**
df = df.drop_duplicates(subset="decrypted_id", keep="first")

# 2️⃣ **Manejo de datos faltantes en "school"** (Forzar reemplazo correctamente)
if "school" in df.columns:
    df["school"] = df["school"].replace("", "Desconocido").fillna("Desconocido")

# 3️⃣ **Normalización y Estandarización: Redondear "G1"**
if "G1" in df.columns:
    df["G1"] = df["G1"].round(0)

# Promedio de G1, G2 y G3
df["G-Average"] = ((df["G1"] + df["G2"] + df["G3"]) / 3).round(0)

### 📂 **Exportar Datos Limpios** ###
df.to_csv("datos_limpios.csv", index=False)  # Guardar en CSV

# 🔍 **Mostrar los datos transformados**
###tools.display_dataframe_to_user(name="Datos Limpiados", dataframe=df)

print("✅ Proceso finalizado. Datos limpios guardados en 'datos_limpios.csv'.")

### MODELO RANDOM FOREST
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

### 🔹 **1. Cargar Datos desde CSV** ###
file_path = r"C:\Users\ledez\Desktop\datos_limpios.csv"  # Ruta del CSV
df = pd.read_csv(file_path)

### 🔹 **2. Preparar los Datos** ###
# Verificar qué columnas son categóricas
categorical_columns = df.select_dtypes(include=["object"]).columns.tolist()

# Convertir variables categóricas en numéricas (one-hot encoding)
df_encoded = pd.get_dummies(df, columns=categorical_columns, drop_first=True)

# Lista de variables predictoras (excluyendo "decrypted_id", "G1", "G2", "G3", "G-Average")
features = [col for col in df_encoded.columns if col not in ["G1", "G2", "G3", "G-Average", "decrypted_id"]]

# Definir X (variables predictoras) e y (variable objetivo)
X = df_encoded[features]
y = df_encoded["G-Average"]

# Asegurar que todas las variables en X son numéricas
X = X.apply(pd.to_numeric, errors="coerce")

# Dividir los datos en conjunto de entrenamiento (80%) y prueba (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

### 🔹 **3. Entrenar el Modelo Random Forest** ###
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

### 🔹 **4. Evaluación del Modelo** ###
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\n📊 **Evaluación del Modelo:**")
print(f"🔹 MAE (Error Absoluto Medio): {mae:.2f}")
print(f"🔹 MSE (Error Cuadrático Medio): {mse:.2f}")
print(f"🔹 RMSE (Raíz del Error Cuadrático Medio): {rmse:.2f}")
print(f"🔹 R² (Coeficiente de Determinación): {r2:.2f}\n")

### 🔹 **5. Importancia de las Variables** ###
importances = model.feature_importances_
feature_importance_df = pd.DataFrame({"Feature": X.columns, "Importance": importances})
feature_importance_df = feature_importance_df.sort_values(by="Importance", ascending=False)

# Mostrar los resultados
print("📌 **Importancia de las Variables en el Modelo:**")
print(feature_importance_df.head(10))  # Mostrar las 10 más importantes


