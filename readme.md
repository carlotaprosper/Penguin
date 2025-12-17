# 🐧 Penguin Vision – Flask + IA + ONNX

Esta aplicación web ha sido desarrollada con **Flask**, permite **detectar la especie de un pingüino** a partir de una imagen o datos manuales, **generar estadísticas**, **crear historias emotivas con IA** y **apadrinar pingüinos** mediante un sistema de adopción con certificado.

La app combina **Machine Learning (ONNX)**, **modelos de visión con IA generativa**, **MongoDB** y **visualización de datos**.

---

## 🚀 Funcionalidades Principales

### 📷 Predicción y Visión
- 🎞️ **Predicción por Imagen**:
    - Introduce la URL de una imagen de un pingüino
    - Estimación automática de características físicas usando IA
    - Predicción de especie mediante un modelo Random Forest (ONNX)
- ✍️ **Predicción Manual:** 
    - Introducción manual de medidas del pingüino
    - Clasificación automática de la especie
    - Generación de una representación visual del pingüino usando **Pollination AI**

### 🧠 Modelo ONNX
El núcleo predictivo es un modelo **Random Forest** exportado a formato `.onnx` para máxima portabilidad y velocidad, entrenado para clasificar entre:
- 🐧 **Adelie**
- 🐧 **Chinstrap**
- 🐧 **Gentoo**

### 🗃️ Datos y Persistencia
- 💾 **MongoDB:** 
    - Registro de pingüinos
    - Historial de predicciones
    - Gestión de pagos y adopciones
- 📊 **Dashboard de Estadísticas:** Visualización gráfica con **Matplotlib/Pandas** sobre:
    - Peso medio por especie
    - Tamaño del pico y de las aletas
    - Diferencias por sexo
    - Número de adopciones
    - Visualización con gráficos

### 💖 Apadrinamiento y Narrativa
- **IA**:
    - Historias emotivas generadas por IA
    - Sistema de adopción
    - Certificado personalizado

---

## 🧠 Tecnologías Utilizadas

| Área | Tecnologías |
| :--- | :--- |
| **Backend** | Python, Flask, Jinja2, Requests |
| **Machine Learning** | ONNX Runtime, Scikit-learn (Entrenamiento previo) |
| **IA Generativa** | **Cohere** (Visión y Texto), **Pollination** (Generación de imagen) |
| **Base de Datos** | MongoDB (PyMongo) |
| **Visualización** | Pandas, NumPy, Matplotlib |
| **Utilidades** | Python-dotenv |

---

## 📁 Estructura del Proyecto

```text
├── app.py                      # Punto de entrada de la aplicación Flask
├── penguins_rf.onnx            # Modelo de ML entrenado y exportado
├── requirements.txt            # Dependencias del proyecto
├── .env                        # Variables de entorno (poner en el .gitignore)
├── .gitignore                  # Archivos ignorados por Git
├── training_penguins_pyspark.ipynb # Notebook de entrenamiento (poner en el .gitignore)
├── sounds/                     # Archivos de audio (mp3)
└── templates/                  # Plantillas HTML (Jinja2)
    ├── index.html
    ├── predict.html
    ├── historico.html
    ├── stats.html
    ├── apadrinar.html
    ├── adoptar.html
    └── certificado.html