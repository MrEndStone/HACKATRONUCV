import nltk
from flask import Flask, request, jsonify
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

app = Flask(__name__)

# Datos de entrenamiento
entrenamiento = [
    ("Hola", "saludo"),
    ("¿Cómo estás?", "saludo"),
    ("¿Qué hora es?", "tiempo"),
    ("Cuéntame un chiste", "chiste"),
    ("Adiós", "despedida"),
    ("Adios", "despedida"),
]

# Preparar datos de entrenamiento
tokens_entrenamiento = []
intenciones_entrenamiento = []
for frase, intencion in entrenamiento:
    tokens = word_tokenize(frase.lower())
    tokens_entrenamiento.append(' '.join(tokens))
    intenciones_entrenamiento.append(intencion)

# Crear modelo TF-IDF
vectorizador = TfidfVectorizer()
X = vectorizador.fit_transform(tokens_entrenamiento)

# Entrenar clasificador SVM lineal
clasificador = LinearSVC()
clasificador.fit(X, intenciones_entrenamiento)

# Función para clasificar la intención de una pregunta
def clasificar_intencion(pregunta):
    tokens = word_tokenize(pregunta.lower())
    vector = vectorizador.transform([' '.join(tokens)])
    intencion = clasificador.predict(vector)[0]
    return intencion

# Ruta para el servicio web
@app.route('/chatbot', methods=['POST'])
def chatbot():
    pregunta = request.json['pregunta']
    if pregunta.lower() == "adiós":
        respuesta = "¡Hasta luego! Fue un placer ayudarte."
    else:
        intencion = clasificar_intencion(pregunta)
        if intencion == "saludo":
            respuesta = "¡Hola! ¿En qué puedo ayudarte?"
        elif intencion == "tiempo":
            respuesta = "Lo siento, no puedo decirte la hora en este momento."
        elif intencion == "chiste":
            respuesta = "¿Por qué los pájaros no usan Facebook? Porque ya tienen Twitter."
        else:
            respuesta = "Lo siento, no entendí tu pregunta."
    
    return jsonify({'respuesta': respuesta})

if __name__ == "__main__":
    nltk.download('punkt')  # Descargar recursos de NLTK
    app.run()