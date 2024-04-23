import streamlit as st
import pandas as pd 
from sklearn.preprocessing import OrdinalEncoder
from sklearn.neighbors  import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("ejalisco2016.csv", index_col= "CLAVE DE INMUEBLE")
df_train = pd.read_csv("train.csv", index_col= "CLAVE DE INMUEBLE")
df_test = pd.read_csv("test.csv", index_col= "CLAVE DE INMUEBLE")

st.set_page_config( page_title="Jalisco Connection Points")


st.title("Jalisco Connection Points :globe_with_meridians:")

columna_texto, columna_imagen = st.columns([2, 1])

with columna_texto:
    st.markdown("""
    <span style='line-height: 0.8;'>
    Esteban Javier Berumen Nieto\n
    ITESO  (Instituto Tecnológico y de Estudios Superiores de Occidente)\n
    10 de abril del 2024
    </span>
    """, unsafe_allow_html=True)
with columna_imagen:
    st.image("ITESO.png", use_column_width=True)

st.markdown("""
#### Introducción

En este proyecto, se emplea el algoritmo KNN para predecir el ancho de banda 
contratado en 2014, por un listado de instituciones públicas de Jalisco en el año 2016. El 
objetivo principal es evaluar la precisión del modelo en función de las características de 
estas instituciones. 
            
Se tomo la decision de utilizar el algoritmo de knn dado que como veremos mas adelante la 
distrubucion de los datos no es normal.

Los datos fueron obtenidos del sitio [Datos Abiertos Jalisco](https://datos.jalisco.gob.mx/dataset/puntos-de-conexion-ejalisco) en donde encontramos un dataset con las siguientes variables incluidas:
**[clave de inmueble, tecnologia instalada, ancho de banda contratado 2014, institucion, 
nombre del centro, turno/horario, nivel, region, municipio, localidad, domicilio, 
codigo postal, longitud, latitud]** en donde encontramos 6716 registros.
""")

st.write(df)

st.markdown("""
#### Desarrollo

Dentro del preprocesamiento de datos se realizó una limpieza en donde todos los 
datos faltantes y N/D del dataset fueron remplazados por la primera moda de la columna 
correspondiente, esto ya que todas las columnas del dataset son categóricas, esto a su vez 
hiso que fuera necesario usar el ***OrdinalEncoder*** de la librería ***sklearn***, para poder realizar la 
codificación de las variables, además se dividió el data set e un train y test en donde el train 
es el **20%** del original y test es el **80%**.

""")

st.markdown("""
##### Train data :bar_chart:
""")
st.write(df_train)

st.markdown("""
##### Test data :bar_chart:
""")
st.write(df_test)

st.markdown( """
A continuación, se eliminaron algunas columnas debido de diferentes problemas u 
objetivos; como lo son las columnas de: **clave de inmueble, ancho de banda contratado 
2014, nombre del centro, longitud, latitud**. En el caso del ancho de banda, esta columna 
se convierte en el target. También se realizaron los histogramas de las diferentes columnas, 
así como un mapa de correlación de las variables. 
""")

st.image("histograms_x_train.png")
st.image("histograms_x_test.png")
st.image("histograms_y_train.png")
st.image("histograms_y_test.png")

st.markdown("""
Como ya meciono anterior mente en ninuguno de los features encontramos una distribucion normal,
gracias a esto tambien podemos ver que en entre el train y el test hay una gransimetria.

Algunas otras cosas que podemos ver en algunos features espesificos son:
La mayoria de los regitros de Horario corresponde a solo un horaio (Matutino)

""")

st.image("Corr_X_train.png")
st.image("Corr_X_test.png")

st.markdown("""
Se implemento el algoritmo knn en el cual se probó con diferentes k para poder 
establecer cuál sería el número de vecinos mediante la prueba del codo.
""")

st.image("knn_scores.png")

#!----------------------------------------------------------------
X_train = pd.read_csv("X_train.csv")
y_train = pd.read_csv("y_train.csv")
X_test = pd.read_csv("X_test.csv")
y_test = pd.read_csv("y_test.csv")

@st.cache_data
def train_knn_model(X_train, y_train, n_neighbors):
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(X_train, y_train)
    return knn

# Barra lateral para ajustar el número de vecinos
n_neighbors = st.slider('Número de vecinos (k)', min_value=1, max_value=30, value=5, step=1)

# Reentrenar el modelo con los nuevos parámetros
knn_model = train_knn_model(X_train, y_train, n_neighbors)

# Calcular precisión del modelo actualizado
y_pred = knn_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Mostrar precisión del modelo
st.write(f'Precisión del modelo KNN con **{n_neighbors}** vecinos: **{accuracy:.5f}**')
#!----------------------------------------------------------------

st.markdown("""
#### Resultados  
Visto lo anterior podemos decir que el algoritmo llega su mejor precisión con 23 
vecinos en donde nos un 84.8% de precisión
""")

code = """
from sklearn.neighbors  import KNeighborsClassifier
clf = KNeighborsClassifier(n_neighbors= 23 )
clf.fit(X_train,y_train)
clf.score(X_test,y_test)
"""
st.code(code, language='python')
st.markdown("0.846441947565543")

st.markdown("""
#### Conclusiones  
Al intentar predecir el ancho de banda que debería de tener una institución pública 
en Jalisco basándonos en diferentes características, el algoritmo knn nos da una precisión 
de hasta el 84.8% utilizando 23 vecinos, además de esto como vimos en los mapas de 
correlación la mayoría de las variables no tiene no gran correlación entre si, sin embargo 
vemos que las que tiene más correlación con el ancho de banda son la tecnología instalada 
y el estatus, por lo que podríamos decir que ancho  de banda depende u poco mas de estos 
que de las demás características.
""")

st.markdown("""
#### Referencias  
Secretaría de Educación Pública de Jalisco. (s/f). Puntos de Conexión eJalisco [Conjunto de 
datos]. Recuperado de https://datos.jalisco.gob.mx/dataset/puntos-de-conexion-ejalisco  
scikit-learn. (s/f). Nearest Neighbors. Recuperado de https://scikit-
learn.org/stable/modules/neighbors.html 
""")




