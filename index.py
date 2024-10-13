# Importamos las librerías necesarias
import pandas as pd  # Para manipulación de datos
import numpy as np  # Para generación de datos aleatorios
from sklearn.model_selection import train_test_split  # División del dataset
from sklearn.preprocessing import LabelEncoder  # Convertir variables categóricas
from sklearn.tree import DecisionTreeClassifier, plot_tree  # Modelo y visualización
from sklearn.metrics import classification_report, confusion_matrix  # Evaluación del modelo
import matplotlib
import matplotlib.pyplot as plt  # Para mostrar y guardar gráficos

# Usamos el backend 'Agg' para evitar problemas con Tkinter
matplotlib.use('Agg')

# Configuramos una semilla para reproducibilidad
np.random.seed(42)

# Creamos un DataFrame con datos simulados de transporte masivo
data = pd.DataFrame({
    'fecha': pd.date_range(start='2024-01-01', periods=20),
    'hora_pico': np.random.randint(0, 2, 20),
    'dia_semana': np.random.choice(['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes'], 20),
    'num_pasajeros': np.random.randint(20, 300, 20),
    'demanda_esperada': np.random.choice(['Alta', 'Media', 'Baja'], 20),
    'clima': np.random.choice(['Soleado', 'Lluvioso', 'Nublado'], 20),
    'accidentes': np.random.randint(0, 5, 20)
})

# Guardamos el DataFrame como JSON y lo cargamos de nuevo
data.to_json('transporte_masivo.json', orient='records', date_format='iso', indent=2)
data = pd.read_json('transporte_masivo.json')

# Convertimos variables categóricas a numéricas
le = LabelEncoder()
data['dia_semana'] = le.fit_transform(data['dia_semana'])
data['demanda_esperada'] = le.fit_transform(data['demanda_esperada'])
data['clima'] = le.fit_transform(data['clima'])

# Definimos variables independientes (X) y dependiente (y)
X = data[['hora_pico', 'dia_semana', 'num_pasajeros', 'clima', 'accidentes']]
y = data['demanda_esperada']

# Dividimos los datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Creamos y entrenamos el modelo de Árbol de Decisión
modelo = DecisionTreeClassifier(random_state=42)
modelo.fit(X_train, y_train)

# Hacemos predicciones y evaluamos el modelo
y_pred = modelo.predict(X_test)
print("Matriz de Confusión:")
print(confusion_matrix(y_test, y_pred))
print("\nReporte de Clasificación:")
print(classification_report(y_test, y_pred))

# Configuramos el gráfico
plt.figure(figsize=(15, 8))
plot_tree(modelo, feature_names=X.columns, class_names=['Alta', 'Media', 'Baja'], filled=True)

# Guardamos el gráfico como PNG
plt.savefig('arbol_decision.png')
print("El gráfico del árbol de decisión se ha guardado como 'arbol_decision.png'.")
