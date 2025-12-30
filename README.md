# Proyecto-de-ML-y-RBI
Este proyecto para las materias Aprendizaje Automático y Razonamiento Bajo Incertidumbre muestra una comparativa entre 3 algoritmos de búsqueda Hill climbing, Recocido Simulado y un Híbrido para aprender una red Bayesiana a partir de una base de datos guiandose de la métrica BIC.

Pasos para ejecutar correctamente los códigos:

Paso 1: Si tu base de datos esta discretizada la puedes usar directamente, de no ser así ingresarla en el algoritmo CAIM.py en formato csv modificando las siguientes líneas.
mi_archivo = "iris.csv" #Colocar el nombre de la base de datos.
mi_columna_clase = "clase" #Colocar el nombre de la columna objetivo o clase.

Paso 2:Una vez con la base de datos ya discretizada ejecutar el archivo PROYECTO_ML_RBI.py cambiando las siguientes líneas.
mis_archivos = ["ejemplo_Iris_Discretizado.csv"] #Introducir el nombre de tu archivo.
TARGET_COL = 'target' #Introducir el nombre de la columna objetivo o clase.
K_FOLDS = 5 #Elegir el número de folds para la validación cruzada.

Paso 3: Al final obtendrás un comparativa en forma de gráfico de cada red bayesiana, su score BIC, precisión y el tiempo que tardo en ejecutarse cada algoritmo. De igual manera 3 archivos en formato xml que sirven para ingresalor a Weka directamente y realizar pruebas con la red bayesiana final y la base de datos.

NOTA IMPORTANTE: Para que el codigo funcione directamente debe tener las bases de datos en la misma carpeta de los códigos, de no ser así debe copiar la ruta exacta de los archivos.
