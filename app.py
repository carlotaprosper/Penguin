from flask import Flask, render_template, request
import cohere
import json
import os
import numpy as np
import onnxruntime as ort
import requests
from urllib.parse import quote
from dotenv import load_dotenv

# leemos el .env
from dotenv import load_dotenv
load_dotenv() # se le pone la ruta al .env

# leer variable de entorno cohere_api_key
cohere_api_key = os.getenv("COHERE_API_KEY")

co = cohere.ClientV2(api_key=cohere_api_key)

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "penguins_rf.onnx")

# Cargar el modelo ONNX
onnx_session = ort.InferenceSession(
    "penguins_rf.onnx",
    providers=["CPUExecutionProvider"]
)

# Tomamos el primer input y la primera salida del modelo ONNX
onnx_input_name = onnx_session.get_inputs()[0].name
onnx_output_name = onnx_session.get_outputs()[0].name

# Mapeo de etiqueta numérica -> nombre de especie
LABEL_TO_SPECIES = {
    0: "Adelie",
    1: "Chinstrap",
    2: "Gentoo",
}

def predict_species_rf(
    features
):
    """
    Devuelve (label_int, species_str) para un pingüino con las features dadas.

    El orden de las features debe ser el mismo que en el entrenamiento:
        [bill_length_mm, bill_depth_mm, flipper_length_mm, body_mass_g, sex_num]
    """
    x = np.array([[
        float(features["bill_length_mm"]),
        float(features["bill_depth_mm"]),
        float(features["flipper_length_mm"]),
        float(features["body_mass_g"]),
        float(features["sex_num"])
    ]], dtype=np.float32)

    # Ejecutar inferencia ONNX
    outputs = onnx_session.run([onnx_output_name], {onnx_input_name: x})
    # Normalmente la primera salida es la etiqueta predicha
    label_int = int(outputs[0][0])
    species = LABEL_TO_SPECIES.get(label_int, f"desconocida ({label_int})")
    return label_int, species

# cambiar label/specie
def get_features_from_image(img_url):
    prompt = """Eres un modelo que recibe una imagen de entrada y tu ÚNICA tarea es devolver SIEMPRE un JSON con cinco campos numéricos: "bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g" y "sex_num".
    INSTRUCCIONES IMPORTANTES:
    1. FORMATO DE SALIDA
    - Debes responder SIEMPRE y SOLO con un objeto JSON válido, sin texto adicional antes ni después.
    - El formato exacto debe ser:
   {
        "bill_length_mm": <float>,
        "bill_depth_mm": <float>,
        "flipper_length_mm": <float>,
        "body_mass_g": <float>,
        "sex_num": <float 0.0 o 1.0>
    }
    - No incluyas comentarios, explicaciones, texto libre ni ningún otro campo.
    - Usa SIEMPRE comillas dobles en las claves ("bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g", "sex_num").
    - Los cinco valores deben ser números float (sin comillas).
    2. SIGNIFICADO DE LOS CAMPOS
    - "bill_length_mm": float que representa la longitud del pico en milímetros del pingüino principal de la imagen.
    - "bill_depth_mm": float que representa la profundidad del pico en milímetros del pingüino principal de la imagen.
    - "flipper_length_mm": float que representa la longitud del pico en milímetros del pingüino principal de la imagen.
    - "body_mass_g": float que representa el peso de del pingüino principal de la imagen en gramos.
    - "sex_num": float que representa el sexo del pingüino principal de la imagen.
        * 0.0 = hembra
        * 1.0 = macho
    3. CUÁNDO HAY UN PINGÜINO CLARO EN LA IMAGEN
    - Si ves claramente un pingüino que parece ser el protagonista de la imagen:
        * Estima sus features ("bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g") según lo que más probable te parezca.
        * Estima su sexo ("sex_num") como un float según lo que más probable te parezca(0.0=hembra, 1.0=macho).
    4. CUÁNDO HAY VARIOS PINGÜINOS, NINGUNO O ES MUY DIFÍCIL
    - Si hay varios pingüinos:
        * Escoge un pingüino que parezca protagonista (por ejemplo, el más centrado o más cercano a la cámara) y decide los cinco valores para ese pingüino.
    - Si no se ve claramente ningún pingüino o la imagen es demasiado confusa:
        * INVÉNTATE valores razonables para "bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g" y "sex_num".
    - Si tienes dudas sobre cualquiera de los campos:
        * AUN ASÍ debes elegir un valor y devolverlo. Nunca uses null, ni -1, ni dejes el campo vacío.
    5. REGLA MÁS IMPORTANTE
    - PASE LO QUE PASE debes devolver SIEMPRE un JSON con este esquema exacto:
    {
        "bill_length_mm": <float>,
        "bill_depth_mm": <float>,
        "flipper_length_mm": <float>,
        "body_mass_g": <float>,
        "sex_num": <float 0.0 o 1.0>
    }
    - No añadas más campos.
    - No devuelvas texto adicional.
    - No expliques tus decisiones.
    - Aunque no estés seguro, elige los valores más razonables que puedas o invéntalos."""

    response = co.chat(
        model="command-a-vision-07-2025",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            # Can be either a base64 data URI or a web URL.
                            "url": img_url,
                            "detail": "auto"
                        }
                    }
                ]
            }
        ]
    )
 
    features = json.loads(response.message.content[0].text)


    label_int, species = predict_species_rf(features)

    prediccion = (
                f"Especie detectada: {species} (Clase {label_int}). "
                f"Datos usados: Pico {features['bill_length_mm']}x{features['bill_depth_mm']}mm, "
                f"Aleta {features['flipper_length_mm']}mm, Peso {features['body_mass_g']}g, Sexo {features['sex_num']}."
            )

    return prediccion, features

 
def get_images(features, species):
    f = features
    # Crear prompt
    prompt = f"National geographic photo of a real {species} penguin in Antarctica. "
    mass = float(f["body_mass_g"])
    if mass > 5000: prompt += "The penguin is big and fluffy. "
    elif mass < 3500: prompt += "The penguin is small. "
    prompt += "Hyperrealistic, 8k, snow, cinematic lighting."
    
    # Codificar prompt para URL (cambia espacios por %20, etc.)
    encoded_prompt = quote(prompt)
    
    # Construir URL directa
    # seed=42 asegura consistencia, nologo=true quita marcas de agua si es posible
    image_url = f"https://image.pollinations.ai/prompt/{encoded_prompt}?width=1024&height=1024&nologo=true&seed={int(mass)}"
    
    return image_url
    


@app.route("/", methods = ['GET']) #"/" --> endpoint
def home():
    return render_template("index.html")

@app.route("/predict", methods = ["POST"])
def predict():
    #form es un diccionario
    img_url = request.form.get("img_url")
    features = None
    prediccion = ""
    img_gen = None

    if img_url:
        print(f"Procesando la imagen")
        prediccion, features = get_features_from_image(img_url)
        
    else:
        print("Procesando entrada manual.")
        features = {
                "bill_length_mm": request.form.get("bill_length_mm"),
                "bill_depth_mm": request.form.get("bill_depth_mm"),
                "flipper_length_mm": request.form.get("flipper_length_mm"),
                "body_mass_g": request.form.get("body_mass_g"),
                "sex_num": request.form.get("sex_num")
            }
        
        if features:
            label_int, species = predict_species_rf(features)

            prediccion = (
                f"Especie detectada: {species} (Clase {label_int}). "
                f"Datos usados: Pico {features['bill_length_mm']}x{features['bill_depth_mm']}mm, "
                f"Aleta {features['flipper_length_mm']}mm, Peso {features['body_mass_g']}g, Sexo {features['sex_num']}."
            )
            if not img_gen:
                img_gen = get_images(features, species)

    return render_template("index2.html", prediction= prediccion, features = features, img_url = img_url, img_gen=img_gen)

if __name__ == "__main__":
    app.run(debug = True, host = "localhost", port  = 5000)