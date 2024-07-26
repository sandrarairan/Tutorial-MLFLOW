# Tutorial-MLFLOW
Tutorial MLFLOW
# TUTORIAL MLFLOWS EXPERIMENTO Kmeans

Por Sandra Rairan 

ES UNA PLATAFORMA DE CODIGO ABIERTO QUE AYUDA ADMINISTRAR EL CICLO DE VIDA DEL APRENDIZAJE AUTOMATICO DE EXTTREMO A EXTREMO.

Consta de 4 componentes:

- **MLflow Tracking:** Una interfaz de usuario para registrar parámetros, versiones de código, métricas y archivos de salida al ejecutar el código de aprendizaje automático y para visualizar los resultados más adelante.
- **MLflow Projects:** paquete de codigo ML en una forma reutilizable
- **MLflow Models:** Gestiona e implementa modelos de varias bibliotecas ML
- **MLflow Model Registry:** Administrar el ciclo de vida de los modelos

**Introducción a Experimentos con MLflow**

- Importar Mlflow e iniciar el tracking

```python
import mlflow
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
import logging

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)
```

Esta es la forma más común de ejecutar MLflow, en este caso, MLflow se ejecuta en el mismo host donde se ejecuta el código de Python. Para ejecutar MLflow en localhost, simplemente ejecuta el siguiente comando en la línea de comandos con el ambiente activo:

**`mlflow ui`**

Lo anterior iniciará un servidor web de MLflow en el puerto 5000 (por defecto). Para acceder a la interfaz de usuario de MLflow, abre la dirección que te retorna mlflow en tu navegador. Si presentas problemas con el puerto puedes especificar otro puerto ejecutando el comando

**`mlflow ui --port`**

- También podremos crear una carpeta de forma local e indicarle a mlflow que lo fijamos como *_set_tracking_uri_*
En la celda a continuación te enseñaré cómo hacerlo:

`print(f"tracking URI: '{mlflow.get_tracking_uri()}'")`

```python
mlflow.set_experiment("iris_experiment")
with mlflow.start_run(run_name = "example_1"):

    X, y = load_iris(return_X_y= True)
    params = {"C": 0.1, 
              "random_state": 42}
    mlflow.log_params(params)
    lr = LogisticRegression(**params).fit(X, y)
    y_pred = lr.predict(X)
    mlflow.log_metric("accuracy", accuracy_score(y, y_pred))
    mlflow.sklearn.log_model(lr, "model")
    print(f"default artifact location: '{mlflow.get_artifact_uri()}'")
```

```markdown
Folder diferente a mlruns, se puede? Si, puedes crear una carpeta y especificarle a mlflow que lo fijamos como *_set_tracking_uri_*
. Para abrir el tracking folder en la termina, debemos especificar el path de la carpeta que creamos y ejecutar el siguiente comando:

mlflow ui --backend-store-uri file:////ruta_de_acceso_a_la_carpeta
```

```python
import mlflow
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
import logging

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)
```

mlflow.end_run()

```python
mlflow.set_experiment("iris_experiment_ml_local")
mlflow.set_tracking_uri("/home/sr/mlops/tracking/experiment_ml")

with mlflow.start_run(run_name = "example_1"):

    X, y = load_iris(return_X_y= True)
    params = {"C": 0.1, 
              "random_state": 42}
    mlflow.log_params(params)
    lr = LogisticRegression(**params).fit(X, y)
    y_pred = lr.predict(X)
    mlflow.log_metric("accuracy", accuracy_score(y, y_pred))
    mlflow.sklearn.log_model(lr, "model")
    print(f"default artifact location: '{mlflow.get_artifact_uri()}'")

# en la terminal vamos a estar a nivel de la carpeta "experiment_ml" y ejecutamos lo siguiente   
# mlflow ui --backend-store-uri file:////home/sr/mlops/tracking//experiment_ml
```

- **Escenario 2: MLflow en localhost con SQLite**

```markdown
Muchos usuarios también ejecutan MLflow en sus máquinas locales con una base de datos compatible con SQLAlchemy : SQLite . En este caso, los artefactos se almacenan en el ./mlrunsdirectorio local y las entidades de MLflow se insertan en un archivo de base de datos SQLite mlruns.db. Es bastante similar al escenario 1, pero usamos como back a sqlite. En lo personal adora los dos escenarios, sin embargo, también podemos usar un bucket en aws como back. Más adelante lo veremos. 

Para abrir el tracking folder con db como back, ejecutamos lo siguiente:

mlflow ui --backend-store-uri sqlite:///backend.db
```

```python
mlflow.set_tracking_uri("sqlite:///backend.db")
mlflow.set_experiment("Experimento_3")

with mlflow.start_run(run_name = "example_1"):

    X,y = load_iris(return_X_y= True)
    params = {"C": 0.1, "random_state": 42}
    mlflow.log_params(params)

    lr = LogisticRegression(**params).fit(X, y)
    y_pred = lr.predict(X)
    mlflow.log_metric("accuracy", accuracy_score(y, y_pred))
    mlflow.sklearn.log_model(lr, artifact_path="models")
    print(f"default artifacts URI: '{mlflow.get_artifact_uri()}'")
```

Nota: los escenarios de ejecutar Mlflow fueron realizados en el curso de MLODS de Platzi

**Experimento clustering**

```python
from sklearn.preprocessing import StandardScaler
# Unskew the data
data_log = np.log(df_data)

# Initialize a standard scaler and fit it
scaler = StandardScaler()
scaler.fit(data_log)

# Scale and center the data
data_normalized = scaler.transform(data_log)

# Create a pandas DataFrame
data_normalized = pd.DataFrame(data=data_normalized, index=df_data.index, columns=df_data.columns)
data_normalized
```

```python
#Element 4: Dependencies 
# Import KMeans
from sklearn.cluster import KMeans

# Initialize KMeans
kmeans = KMeans(n_clusters=3, random_state=1)

# Fit k-means clustering on the normalized data set
kmeans.fit(rfm_normalized)

# Extract cluster labels
cluster_labels = kmeans.labels_
# Hacer predicciones en el conjunto de prueba
y_pred = kmeans.predict(X_test_scaled)
```

**Definir la firma del modelo (entradas y salidas)**

```python
from mlflow.models.signature import infer_signature
signature = infer_signature(data_normalized, kmeans.predict(data_normalized))
```

**Proveer un ejemplo de datos de entrada**

```python
input_example = pd.DataFrame(data=data_normalized, index=df_data.index, columns=df_data.columns)
```

**Configuración del Experimento**

```python
import mlflow
import mlflow.sklearn
from sklearn.preprocessing import StandardScaler
from mlflow.models.signature import infer_signature
```

```python
from mlflow.models.signature import infer_signature
signature = infer_signature(data_normalized, y)
```

```python
# Crear o establecer un experimento
experiment_name = "cluster_kmeans"
mlflow.set_experiment(experiment_clustering)
```

```python
input_example = pd.DataFrame(data=data_normalized, index=df_data.index, columns=df_data.columns)
```

```python
# Iniciar un run en MLflow
with mlflow.start_run() as run:
    # 1. The Heart of It All — Model Binary
     mlflow.sklearn.log_model(
        sk_model=kmeans, #Element 1: The Heart of It All — Model Binary
        artifact_path="modelkmeans",
        signature=signature, #Element 6: Signature
        input_example=input_example #Element 7: Input Example
    )
    
    # 2. Additional Files — Auxiliary Binaries
    # Registrar el scaler como archivo auxiliar
     mlflow.sklearn.log_model(scaler, "scaler") #Element 2: Additional Files
              
    # 5. Metadata — The Model’s Story
     mlflow.log_param("n_clusters", 3)
     mlflow.log_param("random_state", 42)
    
    # Registrar métrica de ejemplo (inercia)
     inertia = kmeans.inertia_
     mlflow.log_metric("inertia", inertia)

     print(f"Modelo registrado en la corrida de MLflow con ID: {run.info.run_id}")
     
```

Terminal

- activar el ambiente virtual
- mlflow ui
