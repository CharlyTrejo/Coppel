![Coppel_logo](https://github.com/CharlyTrejo/Coppel/blob/main/assets/coppel_logo.png)

# **Simulación de Casos con Machine Learning para CENIC (Centro de Investigación Coppel)**

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







## *Ejemplo clásico*




* Standard Scaler: 

A cada dato se le resta la media de la variable y se le divide por la desviación típica:<br>


