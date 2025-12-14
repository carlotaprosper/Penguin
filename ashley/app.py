from flask import Flask, render_template, request
from dotenv import load_dotenv
import json
import pickle
import pandas as pd
import os
from urllib.parse import urlparse, parse_qs, quote_plus
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from io import BytesIO
import base64
load_dotenv() #leer el .env


user = os.getenv("MONGO_USERNAME")
pw = os.getenv("PASSWORD")
app = Flask(__name__)

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
#función para pasar figuras a imagen        
def fig_to_base64(fig):
        buf = BytesIO()
        fig.savefig(buf, format="png")
        buf.seek(0)
        img_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        plt.close(fig)
        return img_base64

#cambiar ruta a "/historico"
@app.route('/historico', methods=['GET'])
def historico():
    collection_penguins = db['Pinguinos']
    history = list(collection_penguins.find({}, {"_id": 0, "img_url": 0, "story":0}))
    return render_template('historico.html', history=history)

@app.route('/stats', methods=['GET'])
def stats():
    collection_penguins = db['Pinguinos']
    history = list(collection_penguins.find({}, {"_id": 0, "img_url": 0, "story":0}))
    df = pd.DataFrame(history)

    #transformación y limpieza de datos 
    df["species"] = df["species"].astype(str)
    df["body_mass_g"] = df["body_mass_g"].astype(float)
    df["sex_num"] = df["sex_num"].astype(float)
    df["bill_length_mm"] = df["bill_length_mm"].astype(float)
    df["bill_depth_mm"] = df["bill_depth_mm"].astype(float)
    df["flipper_length_mm"] = df["flipper_length_mm"].astype(float)
    df["adopt"] = df["adopt"].astype(int)
    df = df.dropna(subset=["sex_num", "body_mass_g", "species", "bill_length_mm", "bill_depth_mm"])

    #peso medio por especie
    mean_species = (df.groupby("species", as_index=False)["body_mass_g"].mean())
    max_value = mean_species["body_mass_g"].max()
    
    #azul clarito para el máximo, azul marino para el resto
    colors = ["lightblue" if value == max_value else "navy" for value in mean_species["body_mass_g"]]

    #Gráfico de barras
    fig1, ax1 = plt.subplots(figsize=(5, 4))

    ax1.bar(
        mean_species["species"].tolist(),
        mean_species["body_mass_g"].tolist(),
        color=colors
    )
    ax1.set_xlabel("Especie")
    ax1.set_ylabel("Peso medio (g)")

    img1 = fig_to_base64(fig1)

    #diagrama horizontal de barras entre especie y tamaño del pico
    mean_bill_species = (df.groupby("species", as_index=False)["bill_length_mm"].mean())

    #especie con el pico más grande
    max_value = mean_bill_species["bill_length_mm"].max()
    colors = ["lightblue" if value == max_value else "navy"for value in mean_bill_species["bill_length_mm"]]
    #gráfico
    fig2, ax2 = plt.subplots(figsize=(6, 4))

    ax2.barh(
        mean_bill_species["species"].tolist(),
        mean_bill_species["bill_length_mm"].tolist(),
        color=colors
    )
    ax2.set_xlabel("Tamaño del pico (mm)")
    ax2.set_ylabel("Especie")

    img2 = fig_to_base64(fig2)

    #gráfica de correlación entre sexo y tamaño del ala
   #tamaño medio del ala por sexo
    mean_flipper = df.groupby("sex_num")["flipper_length_mm"].mean()

    #listas para el gráfico
    x_labels = ["Hembra", "Macho"]
    y_values = [mean_flipper[0.0], mean_flipper[1.0]]
    colors = ["pink", "lightblue"]

    #gráfico de barras
    fig3, ax3 = plt.subplots(figsize=(5, 4))
    bars = ax3.bar(x_labels, y_values, color=colors)

    #etiquetas
    ax3.set_xlabel("Sexo")
    ax3.set_ylabel("Tamaño medio de la aleta (mm)")

    img3 = fig_to_base64(fig3)


    #diagrama de barras sobre el número de adopciones por especie
    adopciones = df.groupby("species")["adopt"].sum()
    max_value = adopciones.max()
    colors = ["lightblue" if value == max_value else "navy" for value in adopciones.values]

    fig4, ax4 = plt.subplots(figsize=(5, 4))
    adopciones.plot(kind="bar", ax=ax4, color=colors)

    ax4.set_xticklabels(adopciones.index, rotation=0)  
    ax4.set_xlabel("Especie")  
    ax4.set_ylabel("Nº adopciones")

    img4 = fig_to_base64(fig4)

    #correlación entre el peso y el tamaño del ala de la especie mas pesada
    #filtrado
    avg_weight = df.groupby("species")["body_mass_g"].mean()
    heaviest = avg_weight.idxmax()  # nombre de la especie más pesada
    fat_specie_df = df[df["species"] == heaviest]
    #gráfico
    fig5, ax5 = plt.subplots(figsize=(6, 4))
    ax5.scatter(fat_specie_df["body_mass_g"], fat_specie_df["flipper_length_mm"], color="navy")

    #etiquetas
    ax5.set_xlabel(f"Peso (g) ({heaviest})")
    ax5.set_ylabel("Tamaño de la aleta (mm)")
    img5 = fig_to_base64(fig5)

    #diagrama horizontal de barras entre especie y profundidad del pico
    depth_species = (df.groupby("species", as_index=False)["bill_depth_mm"].mean())

    #especie con el pico más grande
    max_value = depth_species["bill_depth_mm"].max()
    colors = ["lightblue" if value == max_value else "navy"for value in depth_species["bill_depth_mm"]]
    #gráfico
    fig6, ax6 = plt.subplots(figsize=(6, 4))

    ax6.barh(
        depth_species["species"].tolist(),
        depth_species["bill_depth_mm"].tolist(),
        color=colors
    )
    ax6.set_xlabel("Profundidad del pico (mm)")
    ax6.set_ylabel("Especie")

    img6 = fig_to_base64(fig6)

    #renderizar plantilla con imágenes
    return render_template("stats.html",img1 = img1, img2 = img2, img3 = img3, img4 = img4, img5 = img5, img6 = img6)
if __name__ == "__main__":

    app.run(debug = True, host = "localhost", port  = 5000)