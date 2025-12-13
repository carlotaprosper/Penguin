from flask import Flask, render_template, request, redirect, url_for
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from bson.objectid import ObjectId
from urllib.parse import quote
from dotenv import load_dotenv
from datetime import datetime
import onnxruntime as ort
import numpy as np
import pandas as pd
import requests
import cohere
import pickle
import random
import json
import os

load_dotenv() # leer el .env

# leer variable de entorno cohere_api_key
cohere_api_key = os.getenv("COHERE_API_KEY")
mongo_uri = os.getenv("MONGO_URI")

co = cohere.ClientV2(api_key=cohere_api_key)

app = Flask(__name__)

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


def get_story_from_image(img_url):
    prompt = f"""Genera una historia corta y emotiva para apadrinar a un pingüino basándote en la siguiente imagen: {img_url}. La historia debe ser en español y no debe exceder los 300 caracteres.
Asegúrate de que la historia sea atractiva.
Proporciona solo la historia sin ningún otro texto adicional.
No incluyas un nombre para el pingüino, solo la historia.
"""

    try:
        response = co.chat(
            model="command-a-vision-07-2025",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": img_url}}
                    ],
                }
            ],
        )

        # Extract and clean the story
        story = response.message.content[0].text
        if story:
            return story.strip()

        # If Cohere returned an empty story for some reason, fall through to fallback
        raise RuntimeError("Cohere returned empty story")

    except Exception as e:
        # Log the error for debugging, then return a safe fallback story
        print(f"Warning: Cohere failed to generate story for image '{img_url}': {e}")

        fallbacks = [
            "En la costa helada, un pingüino curioso busca compañía; con tu apoyo tendrá un hogar y nuevas aventuras.",
            "Pequeño y valiente, este pingüino sueña con un amigo que lo cuide; apadrínalo y haz su vida más cálida.",
            "Nacido entre la olas y la nieve, este pingüino espera por alguien que crea en su ternura. Tu apadrinamiento lo protegerá siempre."
        ]
        return random.choice(fallbacks)

def get_db():

    client = MongoClient(mongo_uri, server_api=ServerApi('1'))

    try:

        client.admin.command('ping')

        print("Conexión con MongoDB exitosa")

        return client['Flask_ONNX']

    except Exception as e:

        print(f"Error conectando a MongoDB: {e}")

        raise

db = None

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

    return prediccion, features, species

 
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
    
def post_mongo(features, species, adopt=0, img_url=None):
    collection = db['Pinguinos']

    doc = {
        "species": species,
        "bill_length_mm": features['bill_length_mm'],
        "bill_depth_mm": features['bill_depth_mm'],
        "flipper_length_mm": features['flipper_length_mm'],
        "body_mass_g": features['bill_length_mm'],
        "sex_num": features['bill_length_mm'],
        "adopt": adopt,
        "img_url": img_url,
    }
    collection.insert_one(doc)
    print("💾 Guardado en Mongo")

@app.before_request
def connect_db():

    global db

    if db is None:
        
        db = get_db()

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
        prediccion, features, species = get_features_from_image(img_url)
        post_mongo(features, species, adopt=0, img_url=img_url)
        
    else:
        print("Procesando entrada manual.")
        features = {
                "bill_length_mm": float(request.form.get("bill_length_mm")),
                "bill_depth_mm": float(request.form.get("bill_depth_mm")),
                "flipper_length_mm": float(request.form.get("flipper_length_mm")),
                "body_mass_g": float(request.form.get("body_mass_g")),
                "sex_num": float(request.form.get("sex_num"))
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
            post_mongo(features, species, adopt=0, img_url=img_gen)

    return render_template("predict.html", prediction= prediccion, features = features, img_url = img_url, img_gen=img_gen)

@app.route('/apadrinar', methods=['GET'])
def apadrinar():

    collection_penguins = db['Pinguinos']

    penguins = list(collection_penguins.find({}))

    if not penguins:
        return render_template('apadrinar.html', penguins=None)

    for penguin in penguins:

        if 'story' not in penguin or not penguin['story']:

            img_url = penguin['img_url']
            story = get_story_from_image(img_url)

            print(f"Generated story for penguin {penguin['_id']}: {story}")

            # CORRECT - update by _id
            collection_penguins.update_one(
                {'_id': penguin['_id']}, 
                {'$set': {'story': story}}
            )

    # Fetch updated documents
    update_penguins = list(collection_penguins.find({}))

    for penguin in update_penguins:
        
        penguin['_id'] = str(penguin['_id'])
        penguin['id'] = penguin['_id']

    return render_template('apadrinar.html', penguins=update_penguins)

@app.route('/adoptar/<string:id>', methods=['GET', 'POST'])
def adoptar(id):

    collection = db['Pinguinos']

    if request.method == 'GET':
        penguin = collection.find_one({'_id': ObjectId(id)})

        if penguin:
            penguin['_id'] = str(penguin['_id'])

        return render_template('adoptar.html', penguin=penguin)

    if request.method == 'POST':

        penguin = collection.find_one({'_id': ObjectId(id)})

        nombre = request.form.get('nombre')
        email = request.form.get('email')
        tarjeta = request.form.get('tarjeta')
        nombre_penguin = request.form.get('nombre_penguin')

        # validate required fields
        if not nombre or not email or not tarjeta:
            if penguin:
                penguin['_id'] = str(penguin['_id'])
            error = "Por favor completa los campos obligatorios (nombre, email, tarjeta)."
            return render_template('adoptar.html', penguin=penguin, error=error)

        collection_pay = db['Pagos']

        pago = {
            'penguin_id': id,             
            'nombre': nombre,
            'email': email,
            'tarjeta': tarjeta,
            'nombre_penguin': nombre_penguin,
            'created_at': datetime.utcnow()
        }

        try:

            result = collection_pay.insert_one(pago)
            inserted_id = result.inserted_id
            print(f"Pago insertado con _id: {inserted_id}")

        except Exception as e:

            print(f"Error insertando pago en MongoDB: {e}")

            if penguin:
                penguin['_id'] = str(penguin['_id'])

            error = "Ocurrió un error al registrar el pago. Inténtalo de nuevo."
            return render_template('adoptar.html', penguin=penguin, error=error)

        return redirect(url_for('certificado', payment_id=str(inserted_id)))
    
@app.route('/certificado/<string:payment_id>', methods=['GET'])
def certificado(payment_id):

    collection_pay = db['Pagos']

    try:
        pago = collection_pay.find_one({'_id': ObjectId(payment_id)})

    except Exception:
        return "Certificado no encontrado (ID inválido).", 404

    if not pago:
        return "Certificado no encontrado.", 404

    # Try to fetch the penguin (penguin_id stored as string)
    penguin = None

    try:
        penguin = db['Pinguinos'].find_one({'_id': ObjectId(pago.get('penguin_id'))})
    except Exception:
        penguin = None

    # Prepare data for template
    adopter_name = pago.get('nombre')
    adopter_email = pago.get('email')
    penguin_name = pago.get('nombre_penguin') or (penguin.get('species') if penguin else 'Pingüino')
    penguin_img = penguin.get('img_url') if penguin else None
    created_at = pago.get('created_at')

    if created_at:

        issued = created_at.strftime("%d %B %Y")
        
    else:
        issued = datetime.utcnow().strftime("%d %B %Y")

    certificate_data = {
        'adopter_name': adopter_name,
        'adopter_email': adopter_email,
        'penguin_name': penguin_name,
        'penguin_img': penguin_img,
        'issued': issued,
        'payment_id': str(pago.get('_id'))
    }

    db['Pinguinos'].update_one(
        {'_id': ObjectId(pago.get('penguin_id'))},
        {'$set': {'adopt': 1}}
    )

    return render_template('certificado.html', **certificate_data)

if __name__ == "__main__":
    app.run(debug = True, host = "localhost", port  = 5000)