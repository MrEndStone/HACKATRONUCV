import nltk
nltk.download('punkt')

from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

# Datos de entrenamiento
entrenamiento = [
    ("Hola", "saludo"),
    ("¿Cómo estás?", "saludo"),
    ("¿Qué hora es?", "tiempo"),
    ("Cuéntame un chiste", "chiste"),
    ("Adiós", "despedida"),
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

# Función principal del chatbot
def chatbot():
    print("¡Hola! Soy un chatbot. ¿En qué puedo ayudarte hoy?")
    while True:
        pregunta = input("Tú: ")
        if pregunta.lower() == "adiós":
            print("Chatbot: ¡Hasta luego! Fue un placer ayudarte.")
            break
        intencion = clasificar_intencion(pregunta)
        if intencion == "saludo":
            print("Chatbot: ¡Hola! ¿En qué puedo ayudarte?")
        elif intencion == "tiempo":
            print("Chatbot: Lo siento, no puedo decirte la hora en este momento.")
        elif intencion == "chiste":
            print("Chatbot: ¿Por qué los pájaros no usan Facebook? Porque ya tienen Twitter.")
        else:
            print("Chatbot: Lo siento, no entendí tu pregunta.")

# Ejecutar el chatbot
if __name__ == "__main__":
    chatbot()
