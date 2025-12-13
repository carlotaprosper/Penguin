from flask import Flask, render_template, request, redirect, url_for
from datetime import datetime 
from dotenv import load_dotenv 
import cohere
import json
import pickle
import pandas as pd
import os
from urllib.parse import urlparse, parse_qs, quote_plus
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from bson.objectid import ObjectId 
import random

load_dotenv() #leer el .env

api_key = os.getenv("COHERE_API_KEY")
user = os.getenv("MONGO_USERNAME")
pw = os.getenv("PASSWORD")

co = cohere.ClientV2(api_key=api_key)

app = Flask(__name__)

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

    username = quote_plus(user)
    password = quote_plus(pw)

    uri = f"mongodb+srv://{username}:{password}@clustersanti.etewqjo.mongodb.net/?appName=ClusterSanti"
    
    client = MongoClient(uri, server_api=ServerApi('1'))

    try:

        client.admin.command('ping')

        print("Conexión con MongoDB exitosa")

        return client['Flask_ONNX']
    
    except Exception as e:

        print(f"Error conectando a MongoDB: {e}")

        raise

db = None

@app.before_request
def connect_db():

    global db

    if db is None:
        
        db = get_db()

@app.route('/')
def index():
    return "<h1>Todo Ok</h1>"

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




