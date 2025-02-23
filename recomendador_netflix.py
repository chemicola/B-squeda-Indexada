# 1. Importamos las librerías. 
# pandas: Se usa para cargar y manipular datos en formato tabular
# numpy: Es útil para trabajar con matrices y operaciones numéricas.
import pandas as pd
import numpy as np


# 2. Cargar datos
from sklearn.feature_extraction.text import TfidfVectorizer 
# TfidfVectorizer: Es una herramienta para convertir texto en números (vectores) que la computadora puede entender.
from sklearn.metrics.pairwise import cosine_similarity
# cosine_similarity: Es una técnica para medir qué tan similares son dos cosas basadas en sus características numéricas.

datos = pd.read_csv("netflix_titles.csv") # Cargamos nuestro archivo CSV.
print(datos.head()) # Muestra las primeras 5 filas.

# 3. Combinar características relevantes. Crear una función que toma una fila de la tabla 
# y combina varias columnas en una sola cadena de texto.
def combinar_caracteristicas(fila):
    return f"{fila['director']} {fila['cast']} {fila['listed_in']} {fila['description']}"
datos["características"] = datos.apply(combinar_caracteristicas, axis=1) # Representa todas estas características juntas.

# 4. Convertir texto a números.
tfdif = TfidfVectorizer(stop_words="english") # Creamos un objeto que convierte texto en números.
matriz_tdfif = tfdif.fit_transform(datos["características"]) # Convertimos la columna "características" en una matriz de números.

# 5. Calcular la similitud entre películas.
similitud_coseno = cosine_similarity(matriz_tdfif, matriz_tdfif) # Resultado: Una tabla (matriz) donde cada número indica qué tan similares son dos películas.

# 6. Crear función para recomendar películas.
def recomendar_titulo(titulo, similitud_coseno=similitud_coseno):
    # Encontrar el índice del título.
    indice = datos[datos['title'] == titulo].index[0] # uscamos el número de fila (índice) de la película que queremos usar como referencia.
    
    # Obtenemos una lista de todas las películas con su puntuación de similitud respecto a la película de referencia.
    puntuaciones_similitud = list(enumerate(similitud_coseno[indice])) 
    
    # Ordenar las películas de mayor a menor por similitud.
    puntuaciones_similitud = sorted(puntuaciones_similitud, key=lambda x: x[1], reverse=True)
    
    # Obtener los índices de las 5 películas más similares (excluyendo la misma película)
    indices_similares = [i[0] for i in puntuaciones_similitud[1:6]]
    
    # Devolver los títulos recomendados
    return datos['title'].iloc[indices_similares]

# Ejemplo de uso
print(recomendar_titulo("Inception"))