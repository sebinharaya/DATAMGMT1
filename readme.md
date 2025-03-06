# Proyecto: Construcción de un Data Pipeline y Aplicación de IA

## Créditos
Autor: Sebastián Ledezma  
Curso: Administración de Datos  
Universidad: Lead University  
Primer Cuatrimestre de 2025  

## Descripción
Este proyecto implementa un Data Pipeline para la limpieza y transformación de datos, seguido de la aplicación de un modelo de Machine Learning (Random Forest) para predecir el rendimiento de estudiantes. Finalmente, se desarrolla una aplicación web con Streamlit para visualizar los resultados y permitir interacciones con los usuarios.

## Estructura del Proyecto
```
Proyecto
│── app.py              # Aplicación web en Streamlit
│── Caso 1 DATA MGTM.py # Código principal del pipeline
│── README.md           # Documentación del proyecto
│── requirements.txt    # Librerías necesarias
│── data
│   ├── datos_limpios.csv  # Dataset procesado
│── models
│   ├── modelo_random_forest.pkl  # Modelo entrenado (opcional)
```

## Requisitos Previos
Antes de ejecutar el proyecto, asegúrate de tener instaladas las siguientes herramientas:
- Python 3.8+
- pip actualizado
- Librerías necesarias (instalar con requirements.txt)

### Instalar Dependencias
Ejecuta el siguiente comando en la terminal:
```sh
pip install -r requirements.txt
```
Si requirements.txt no está disponible, instalar las librerías manualmente:
```sh
pip install streamlit pandas numpy scikit-learn matplotlib seaborn gspread google-auth google-auth-oauthlib google-auth-httplib2
```

## Ejecución del Proyecto

### Ejecutar el Data Pipeline
Antes de visualizar los resultados en Streamlit, es necesario ejecutar el pipeline de procesamiento de datos:
```sh
python "Caso 1 DATA MGTM.py"
```
Esto generará el archivo datos_limpios.csv que será usado en la aplicación.

### Iniciar la Aplicación en Streamlit
Una vez que los datos estén listos, inicia la aplicación con:
```sh
streamlit run app.py
```
Esto abrirá la aplicación en el navegador para interactuar con los resultados.

## Características de la Aplicación Web
- Visualización del Dataset con los datos procesados.
- Entrenamiento y evaluación del modelo de IA en tiempo real.
- Gráficos interactivos para explorar la importancia de las variables.
- Personalización del diseño con paletas de colores pastel.
- Predicciones en vivo (futuro desarrollo).

## Futuras Mejoras
- [ ] Permitir al usuario ingresar datos para obtener predicciones personalizadas.
- [ ] Agregar visualizaciones adicionales como histogramas y correlaciones.
- [ ] Optimizar el modelo con técnicas de ajuste de hiperparámetros.

Cualquier sugerencia es bienvenida para mejorar el proyecto.

