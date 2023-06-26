![Coppel_logo](https://github.com/CharlyTrejo/Coppel/blob/main/assets/coppel_logo.png)

# **Simulación de Casos con Machine Learning para CENIC (Centro de Investigación Coppel) **

## **Introducción**
En el presente repositorio se presentan soluciones de Machine Learning para posibles casos de negocio pertenecientes al sector de Grupo Coppel: BanCoppel, Afore Coppel y Tiendas Coppel. 

---
## **BanCoppel** - `Clientes que cancelan tarjetas de Crédito`
### `Contexto`
Los directivos de BanCoppel están preocupados porque cada vez más clientes cancelan sus servicios de tarjeta de crédito. 
Desearían una herramienta capaz de predecir qué cliente está próximo a cancelar su tarjeta de crédito y así de forma proactiva ofrecer al cliente mejores servicios aumentando la posibilidad de que el cliente decida no cancelar su tarjeta de crédito.
### `Obtención de datos`
Para ello se obtiene un dataset con aproximadamente 10,000 usuarios de tarjetas de crédito BanCoppel con los siguientes atributos del cliente:
* CLIENTNUM: Número de cliente
* Attrition_Flag: Estatus (Vigente o abandono de servicios)
* Customer_Age: Edad en años
* Gender: Género 
* Dependent_count: Número de dependientes económicos
* Education_level: Nivel educativo
* Marital_Status: Estado civil
* Income_Category: Nivel de ingresos
* Card_Category: Tipo de tarjeta de Crédito (Producto)
* Credit_Limit : Límite de crédito
* Avg_Open_To_Buy: Diferencia entre límite de crédito asignado y el balance actual en la cuenta.
* Total_Trans_Amt: Suma de transacciones en el último año
* Total_Revolving_Bal: Saldo vencido
* Months_Inactive_12_mon: Número de meses de inactividad en el último año
* Contacts_Count_12_mon : Número de contactos en el último año
* Total_Ct_Chng_Q4_Q1: Diferencia de transacciones (Q4 sobre Q1)


**El procedimiento se encuentra disponible en: [Credit_Card_ChurnCustomers.ipynb](https://github.com/CharlyTrejo/Coppel/blob/main/CreditCard_ChurnCustomers.ipynb)** 

### `Exploración y Análiside Datos`
El dataset cuenta con 10,127 registros, de los cuales no hay datos nulos, ni duplicados. 

De los 10,127 registros 8,500 pertenecen a clientes activos, mientras que sólo 1,627 son de clientes que han cancelado su tarjeta de crédito, representando el 16.06% de los datos por lo que nos enfrentamos a un dataset con desbalanceo de categorías. 

Mediante un gráfico de correlación se evidencian las variables explicación con mayor asociación (positiva y negativa) con la variable objetivo. 
#### Top 2 variables con fuerte asociación positiva: 
1. Contacts_Count_12_mon
2. Months_Inactive_12_mon
#### Top 3 variables con fuerte asociación negativa:
1. Total_Trans_Ct
2. Total_Ct_Chng_Q4_Q1
3. Total Revolving_Bal

### `Transformación de Datos`
Dato que las columnas "Naive_bayes" no aportan datos relevantes, así como el número de cliente es un dato de entrada irrelevante para un modelo de Machine Learning fueron eliminadas del dataframe.

El dataframe presenta importantes outliers en los atributos Credit_Limit, Avg_Opent_to_Buy y Total_Trans_Amt, dado que el número de registros es relativamente reducido reemplazaré sus datos atípicos con los valores dentro de los límites superiores máximos.

Transformo la variable objetivo con '0' para evento no exitoso (cliente activo) y '1' para evento exitoso (cliente cancela su tarjeta de crédito). 

Realizo una conversión de variables categóricas a varibles dummies mediante método de pandas get_dummies(). 

Además realicé una técnica de escalado a los datos utilizando la clase StandardScale() de scikit-learn. para asegurarme de que las variables tengan una escala similar y estén centradas alrededor de cero; justificado en que los algoritmos  de descenso de gradiente son sensibles a las diferencias en escala.

### `Elección de Modelo`
Elegí trabajar con un Extreme Gradient Boosting (XGBoost) dado su rendimiento, optimización eficiente, manejo de datasets con alta dimensionalidad, flexibilidad en tipos de datos (categóricos y numéricos), es altamente escalable en caso de que sea requerido por la naturalidad del negocio y también por el tema de desbalanceo de datos , ya que XGBoost tiene técnicas incorporadas para manejar problemas de desequilibrio de clases en conjuntos de datos desbalanceados. 

El XGBoost es un algoritmo "Gradient Boosting" basado en árboles de decisión; la idea detrás del Gradient Boosting es entrenar modelos de manera secuencial donde cada modelo se construye para corregir los errores cometidos por el modelo anterior. Es decir, se enfoca en aprender de los errores y mejorar iterativamente el rendimiento del modelo. 
 
### `Performance del Modelo`
--- 
 `Deep Learning`.

Como podemos observar en la imagen supra, ML se inserta en una disciplina más general, macro, conocida como **Inteligencia Artificial**. Para profundizar un poco en estos conceptos, recomendamos acceder al video que se deja a continuación.






## *Ejemplo clásico*



- - - 

## **Esquema de ML**


En el siguiente esquema podremos observar las tres grandes ramas en las que se subdivide Machine Learning:

<p align="center">
<img src="https://pbs.twimg.com/media/ETNlvGZUYAI2iF5.jpg:large"   
height="500px">
</p>

Respecto al alcance de este módulo, abordaremos **APRENDIZAJE SUPERVISADO** y **APRENDIZAJE NO SUPERVISADO**.

## **Aprendizaje supervisado**

>$f(X) = y$

Buscamos un modelo ***f*** que permita determinar la salida ***y*** a partir de la entrada ***X***.

En esta función, ***X*** son los atributos -generalmente se denota con mayúscula porque incluye más de una variable- e ***y*** es la etiqueta. 

El aprendizaje supervisado permite modelar la relación entre las características medidas de los datos y alguna etiqueta asociada con ellos.
Es decir, podremos predecir ***y*** para nuevos datos ***X*** de los cuales no conozcamos la salida.

De acuerdo al tipo de etiquetas que asociamos a los datos, el modelo puede realizar dos tipos de tareas:

+ **`Clasificación:`** las etiquetas son categorías. Ejemplo: enfermo/sano, gato/perro/pájaro, spam/no spam.

+ **`Regresión:`** la variable de salida es un valor numérico. Ejemplo: precio, cantidad, temperatura.

## **Aprendizaje no supervisado**

En este caso, se deja que el conjunto de datos hable por sí mismo. Este modelo tiene datos de entrada, pero no se busca una salida en particular. Implicar modelar las características de un conjunto de datos sin referencia a ninguna etiqueta.

La función de este tipo de algoritmos es encontrar patrones de similaridad. Por eso, incluyen tareas como **`clustering`** (agrupación) y **`reducción de dimensionalidad`**. En este último, el algoritmo busca representaciones más concisas de los datos. En clustering, busca identificar distintos grupos de datos.

Ejemplo de clustering:

<img src="..\_src\assets\clustering.jpg">
  

- - -

## **Correlación**

Por último, para finalizar la parte teórica de esta clase, abordaremos la temática referente a correlación.

En primer lugar, debemos definir la correlación como `concepto teórico` para, en segundo lugar, definirla como `estadístico` (recordemos que un estadístico es cualquier función de la muestra).

La correlación como concepto teórico puede definirse como la relación estadística que tienen dos variables aleatorias entre sí. Esta puede ser o no causal, ya que podemos estar en presencia de una **correlación espuria**. La correlación espuria nos dirá que las variables están relacionadas, cuando en realidad no existe relación causal directa entre ellas.

En cuanto a la correlación como estadístico, es importante diferenciar dos conceptos:

+ *Covarianza*: valor que indica el grado de variación conjunta de dos variables aleatorias respecto a sus medias. Depende de la escala de los datos. Es por este motivo que se suele emplear la correlación, definida a continuación.

+ *Correlación de Pearson*: es la covarianza dividida por el producto de la desviación estándar de cada variable aleatoria.

La correlación nos sirve para determinar qué variables están relacionadas entre sí, con qué intensidad y si siguen una misma dirección o no. Saber qué variables están correlacionadas nos ayudará a construir mejores modelos predictivos.

La correlación no tiene que ser necesariamente lineal. También podemos estar ante dos variables correlacionadas aunque esta relación no se presente de forma lineal.

Para profundizar sobre los tipos de correlación `Pearson`, `Spearman` y `Kendall`, sugerimos este [artículo](https://www.cienciadedatos.net/documentos/pystats05-correlacion-lineal-python.html#:~:text=La%20correlaci%C3%B3n%20de%20Kendall%20es%20un%20m%C3%A9todo%20no%20param%C3%A9trico%20que,cumple%20la%20condici%C3%B3n%20de%20normalidad).

Cerramos esta clase con la siguiente proposición que representa un escenario al que se enfrentarán en múltiples ocasiones y resulta menester identificar: ***correlación no implica causalidad***.


<p align="center">
<img src="https://miro.medium.com/max/1400/1*ERarZ75RoWF8Vn-_AlEmaA.jpeg"
width="1000px" height="400px">
</p>

- - -

#### Enlace Relacionado:

- [Roca, Genís: “La sociedad digital”, TedxTalks](https://www.youtube.com/watch?v=kMXZbDT5vm0)



<img src="../_src/assets/test_Turing.jpg"  height="200">

* En 1952,  Arthur Samuel escribe el primer programa de ordenador capaz de aprender: un programa que jugaba a las Damas y mejoraba en cada partida.

* En 1956, John McCarthy, Marvin Minsky y Claude Shannon acuñaron el término Inteligencia Artificial durante una conferencia en Dartmouth.<br>
<img src="../_src/assets/ia1.jpg"  height="100">

* A mediados de los años 50, con el invento de los transistores, apareció un gran avance con computadoras más poderosas y confiables, y menos costosas. Abriendo paso a la segunda generación de computadoras.

<img src="../_src/assets/transistor.jpg"  height="100">

* En 1958 Frank Rosenblatt diseña el Perceptrón, la primera red neuronal artificial basado el modelo de neuronas de McCulloch y Pitts de 1943.

<img src="../_src/assets/perceptron.jpg"  height="100">

* A mediados de los años 60, aparecen los sistemas expertos, que predicen la probabilidad de una solución bajo un conjunto de condiciones.

* En 1967, la creación del algoritmo conocido como “vecinos más cercanos” permitió a las computadoras utilizar un reconocimiento de patrones muy básico. Incluso tuvo fines comerciales.

* En los años 1970, con John Henry Holland surgió una de las líneas más prometedoras de la inteligencia artificial: los algoritmos genéticos, son llamados así porque se inspiran en la evolución biológica y su base genético-molecular.

* En 1981, Gerald Dejong plantea el concepto «Aprendizaje Basado en Explicación» (EBL), donde la computadora analiza datos de entrenamiento y crea reglas generales que le permiten descartar datos menos importantes.

* En 1982, con la quinta generación de computadoras, el objetivo era el desarrollo de computadoras que utilizarían inteligencia artificial, mejorando hardware como software, sin obtener los resultados esperados: casos en los que es imposible llevar a cabo una paralelización, no se aprecia mejora alguna, o se pierde rendimiento.

<img src="../_src/assets/evolucion_computacion.jpg"  height="100">

* En 1985, Terry Sejnowski inventa NetTalk, un programa que aprende a pronunciar palabras de la misma manera que lo haría un niño.

* En 1986, McClelland y Rumelhart publican Parallel Distributed Processing (Redes Neuronales).

* En 1997, Gari Kaspárov, campeón mundial de ajedrez, pierde ante la computadora autónoma Deep Blue.

* En la década de los 90s, el Machine Learning ganó popularidad gracias a la intersección de la Informática y la Estadística que dio lugar a enfoques probabilísticos en la IA.
Esto generó un gran cambió al campo del aprendizaje automático de las computadoras, ya que se trabajaría con más datos.

* En este periodo que se comenzó a utilizar esta tecnología en áreas comerciales para la Minería de Datos, software adaptable y aplicaciones web, aprendizaje de texto y aprendizaje de idiomas.

* La llegada del nuevo milenio trajo consigo una explosión en el uso del Machine Learning. Geoffrey Hinton en 2006 acuña el término “Deep Learning”, con el que se explican nuevas arquitecturas de Redes Neuronales profundas que permiten a las computadoras “ver” y distinguir objetos y texto en imágenes y videos.

* También en 2006 se consigue facilitar los cálculos independientes necesarios para renderizar cada píxel en GPUs. Hasta entonces era impensable que los científicos usarán GPUs para su trabajo, pero a partir de ese momento lenguajes de alto nivel como C++ o Python se pueden utilizar para programar complejos cálculos y algoritmos permitiendo programar trabajos en paralelo y con gran cantidad de datos.
<img src="../_src/assets/cpu-gpu.jpg"  height="250">

* En 2011 IBM desarrolló Watson, la computadora ganó una ronda de tres juegos seguidos de Jeopardy!.

* En 2016 Google DeepMind vence en el juego Go a un jugador profesional por 5 partidas a 1. El algoritmo realizó movimientos creativos que no se habían visto hasta el momento. El Go es considerado uno de los juegos de mesa más complejos.

* Hoy existen personas que al dialogar sin saberlo con un chatbot no se percatan de hablar con un programa, de modo tal que se cumple la prueba de Turing como cuando se formuló: “Existirá Inteligencia Artificial cuando no seamos capaces de distinguir entre un ser humano y un programa de computadora en una conversación a ciegas.”

* Las redes neuronales son como una caja negra. Esto quiere decir que cuando la máquina da una solución a un problema, es muy complicado conocer cuáles son sus “razonamientos” para llegar a dicha solución.



### Exploración de los datos

Los datos con los que vamos a estar trabajando, son en definitiva la fuente del conocimiento necesario que debemos adquirir para poder resolver las preguntas que nos hacemos, entonces, es preciso conocer todas sus características, algunas de ellas son:

* Variabilidad.
* Estadística.
* Distribución.
* Rangos.






### Reescalar los datos

Muchos algoritmos funcionan mejor normalizando sus variables de entrada. Lo que en este caso significa comprimir o extender los valores de la variable para que estén en un rango definido. Sin embargo, una mala aplicación de la normalización o una elección descuidada del método de normalización puede arruinar los datos y, con ello, el análisis.

* MinMax Scaler: 

Las entradas se normalizan entre dos límites definidos:<br>
<img src="../_src/assets/min_max_formula.jpg" height="50">

Tener en cuenta que si se reescala un atributo, quizás sea conveniente reescalar otro, debido a que estamos rompiendo la proporcionalidad de los datos:

<img src="../_src/assets/min_max.jpg" height="300">

En 1 originalmente, A estaba más cerca de B, al multiplicar por 100, quedó más cerca de C. En 2 el ruido de la señal se hizo más notorio.

* Standard Scaler: 

A cada dato se le resta la media de la variable y se le divide por la desviación típica:<br>
<img src="../_src/assets/standard_scaler_formula.jpg" height="50">

Si bien puede resultar conveniente en datos que no tienen distribución de probabilidad Gaussiana o Normal debido a que se puede trabajar mejor bajo ese esquema, tanto la media como la desviación típica son muy sensibles a outliers.

