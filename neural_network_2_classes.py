import winsound
import pandas
from keras import layers, models, Input
import matplotlib.pyplot as plt
from keras.src.utils.module_utils import tensorflow
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

# Hiperparámetros
train_percentage = 0.70
n = 2
units = 8
activation = "sigmoid"
learning_rate = 0.002
loss = "categorical_crossentropy"
batch_size = 20
epochs = 800
dropout_rate = 0.01
L2_reg_factor = 0.001


# Lectura de los datos
dataset = pandas.read_excel("Data_two_classes.xlsx")
print(dataset)

# Separación de características (features) y etiquetas (labels)
X = dataset[["Fx1",	"Fy1",	"Fz1",	"Tx1",	"Ty1",	"Tz1",
            "Fx2",	"Fy2",	"Fz2",	"Tx2",	"Ty2",	"Tz2",
            "Fx3",  "Fy3",	"Fz3",	"Tx3",	"Ty3",	"Tz3",
            "Fx4",	"Fy4",	"Fz4",	"Tx4",	"Ty4",	"Tz4",
            "Fx5",	"Fy5",  "Fz5",	"Tx5",	"Ty5",	"Tz5",
            "Fx6",	"Fy6",	"Fz6",	"Tx6",	"Ty6",	"Tz6",
            "Fx7",	"Fy7",	"Fz7",  "Tx7",	"Ty7",	"Tz7",
            "Fx8",	"Fy8",	"Fz8",	"Tx8",	"Ty8",	"Tz8",
            "Fx9",	"Fy9",	"Fz9",	"Tx9",  "Ty9",	"Tz9",
            "Fx10",	"Fy10",	"Fz10",	"Tx10",	"Ty10",	"Tz10",
            "Fx11",	"Fy11",	"Fz11",	"Tx11",	"Ty11", "Tz11",
            "Fx12",	"Fy12",	"Fz12",	"Tx12",	"Ty12",	"Tz12",
            "Fx13",	"Fy13",	"Fz13",	"Tx13",	"Ty13",	"Tz13",
            "Fx14",	"Fy14",	"Fz14",	"Tx14",	"Ty14",	"Tz14",
            "Fx15",	"Fy15",	"Fz15",	"Tx15",	"Ty15",	"Tz15"]]  # Seleccionar características (variables de entrada)
Y = dataset[["No Error", "Error"]]  # Seleccionar etiqueta (variable objetivo)

scaler = StandardScaler()
X_normalized_array = scaler.fit_transform(X)

X_normalized = pandas.DataFrame(X_normalized_array, columns=X.columns)

print("Validacion de datos:")
print(X)
print(Y)


X_train, X_test, y_train, y_test = (train_test_split(X_normalized, Y, test_size=1 - train_percentage, random_state=23))

# Inicialización del modelo
network = models.Sequential()

# Declaración de la capa de entrada
network.add(Input(shape=(90,)))  #entradas

# Ciclo de capas de neuronas intermedias
for i in range(n):  # n: número de neuronas intermedias
    #units1 = int(units/(i+1))
    network.add(layers.Dense(
            units=units,  # units: número de neuronas por capa
            activation=activation))  # activation: función de activación elegida.
            # VER DOCUMENTACIÓN PARA VER LAS POSIBLES OPCIONES A ELEGIR
    network.add(tensorflow.keras.layers.Dropout(dropout_rate))
    network.add(layers.Dense(units=units, activation=activation, kernel_regularizer=tensorflow.keras.regularizers.l2(L2_reg_factor)))

# Declaración de la capa de salida
network.add(layers.Dense(
        units=2,  # Una única salida
        activation="sigmoid"))  # Problema de regresión, por tanto salida dada por sigmoide


network.compile(
        optimizer=tensorflow.keras.optimizers.Adam(  # optimizer: algoritmo de optimización
            learning_rate=learning_rate  # learning_rate: ritmo de aprendizaje

        ),
        loss=loss,
        metrics=['accuracy'])  # loss: función de pérdida

losses = network.fit(x=X_train,  # Características de entrada para el entrenamiento
                     y=y_train,  # Etiquetas para el entrenamiento
                     validation_data=(X_test, y_test),  # Datos para la validación durante el entrenamiento
                     batch_size=batch_size,  # Número de muestras por actualización de gradiente
                     epochs=epochs)  # Número de iteraciones completas a través de los datos de entrenamiento

winsound.Beep(350, 500)  # Aviso auditivo de la finalización del entrenamiento.

# Se extrae el historial de error contra iteraciones de la clase
loss_df = pandas.DataFrame(losses.history)

# Crear la primera figura para la pérdida
plt.figure(figsize=(12, 6))  # Tamaño de la figura (ancho, alto)
plt.subplot(1, 2, 1)  # 1 fila, 2 columnas, primer gráfico
loss_df.loc[:, ['loss', 'val_loss']].plot(ax=plt.gca())  # Graficar la pérdida de entrenamiento y validación
plt.title("Curva de Pérdida")  # Título de la gráfica de pérdida
plt.xlabel("Épocas")  # Etiqueta del eje X para la gráfica de pérdida
plt.ylabel("Pérdida")  # Etiqueta del eje Y para la gráfica de pérdida


plt.subplot(1, 2, 2)  # 1 fila, 2 columnas, segundo gráfico
loss_df.loc[:, ['accuracy', 'val_accuracy']].plot(ax=plt.gca())  # Graficar la precisión de entrenamiento y validación
plt.title("Curva de Precisión")  # Título de la gráfica de precisión
plt.xlabel("Épocas")  # Etiqueta del eje X para la gráfica de precisión
plt.ylabel("Precisión")  # Etiqueta del eje Y para la gráfica de precisión

# Agregar información extra de la RN para saber sus hiperparametros
extra_info =     (f"Units: {units} " f"\nLearning Rate: {learning_rate} " f"\nBatch Size: {batch_size}"
                 f"\nEpochs: {epochs}" f"\nDropout Rate: {dropout_rate}")

plt.figtext(0.9, 0.25, extra_info, wrap=True, horizontalalignment='center', fontsize=12)
plt.tight_layout()  # Asegurar que las subplots no se solapen

# Matriz de confusión
y_pred = network.predict(X_test)
y_pred_classes = y_pred.argmax(axis=1)
y_test_classes = y_test.values.argmax(axis=1)
cm = confusion_matrix(y_test_classes, y_pred_classes)

# Visualización de la matriz de confusión
plt.figure(figsize=(12, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', cbar=False,
            xticklabels=["No Error", "Error"],
            yticklabels=["No Error", "Error"])
plt.xlabel('Predecido')
plt.ylabel('Verdadero')
plt.title('Matriz de confusión')

# Estudio de Sensibilidad "Ceteris Paribus"
percentage_variations = [-8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8]  # Variaciones de -40%, -20%, etc.
sensitivity_results = {}

# Generar nuevas predicciones variando cada característica
for i, column in enumerate(X.columns):
    original_value = X_normalized.iloc[0, i]  # Valor original para la primera instancia
    sensitivity_results[column] = {}

    for variation in percentage_variations:
        # Crear variaciones superiores
        upper_variation = original_value * (1 + variation)

        # Copiar la instancia original y aplicar variaciones
        upper_instance = X_normalized.iloc[0].copy()

        # Aplicar variaciones
        upper_instance[i] = upper_variation

        # Realizar predicciones para las instancias variaciones
        y_upper_pred = network.predict(np.array(upper_instance).reshape(1, -1))  # Obtener probabilidades

        # Almacenar los resultados (almacenamos las probabilidades de la clase)
        sensitivity_results[column][f'{variation * 100}%'] = y_upper_pred[0]  # Obtener las probabilidades

# Visualización de los resultados
for feature, predictions in sensitivity_results.items():
    plt.figure(figsize=(12, 6))  # Crear una figura para cada feature

    # Convertir las probabilidades en un formato adecuado para la gráfica
    for class_index in range(y_upper_pred.shape[1]):  # Iterar sobre cada clase
        class_probabilities = [pred[class_index] for pred in predictions.values()]
        plt.plot(predictions.keys(), class_probabilities, marker='o', linestyle='-', label=f'Clase {class_index}')

    plt.title(f'Sensibilidad de {feature} (Probabilidades)')
    plt.xlabel('Variación')
    plt.ylabel('Probabilidad')
    plt.ylim(0, 1)  # Limitar el eje Y entre 0 y 1 para representar las probabilidades
    plt.legend(title="Clases")  # Añadir una leyenda para las clases

plt.show()  # Mostrar la figura

