import pandas as pd #Para el manejo de datos.
import numpy as np #Operaciones númericas.
import os

#Calcular el valor caim para un atributo.
def calcular_caim(caracteristicas, y, cortes):
    #Limite del intervalo.
    contenedores = [-np.inf] + sorted(cortes) + [np.inf]
    #Hacer el intervalo discreto.
    intervalos = pd.cut(caracteristicas, contenedores=contenedores, include_lowest=True)
    #Obtener la matriz cuanta.
    matriz_cuanta = pd.crosstab(y, intervalos)
    #Calcular el número de intervalos dada la matriz.
    n = len(matriz_cuanta.columns)
    #Inicializar el valor CAIM.
    suma_caim = 0.0
    #Iterar cada intervalo y calcular el CAIM.
    for r in matriz_cuanta.columns:
        datos_columna = matriz_cuanta[r]
        max_r = datos_columna.max()
        M_r = datos_columna.sum()
        if M_r > 0:
            suma_caim += (max_r ** 2) / M_r
    return suma_caim / n #Regresar el CAIM/N como dice la fórmula.

#Encontrar los mejores punto de corte usando CAIM.
def obtener_cortes_atributo(valor_caracteristicas, y):
    #Corter candidatos iniciales.
    valores_unicos = np.sort(valor_caracteristicas.unique())
    #Valores únicos ordenados del atributo.
    B = (valores_unicos[1:] + valores_unicos[:-1]) / 2
    #Inicializar variables necesarias.
    candidatos_disponibles = set(B)
    cortes_actuales = []
    global_caim = 0.0
    num_clases = len(y.unique())
    k = 1
    #Bucle que se va a repetir hasta que CAIM no mejore.
    while True:
        mejor_caim_iteracion = -1.0
        mejor_corte_iteracion = None
        #Evaluar cada corte.
        posibles_cortes = list(candidatos_disponibles - set(cortes_actuales))
        #Si ya no se puede mejorar terminar.
        if not posibles_cortes:
            break
        #Evitamos los cortes ya usados.
        for candidato in posibles_cortes:
            cortes_prueba = cortes_actuales + [candidato]
            valor_caim = calcular_caim(valor_caracteristicas, y, cortes_prueba)
            #Si se añade un corde calcular el CAIM.
            if valor_caim > mejor_caim_iteracion:
                mejor_caim_iteracion = valor_caim
                mejor_corte_iteracion = candidato
        
        if (mejor_caim_iteracion > global_caim) or (k < num_clases):
            #Aceptar corte so mejora CAIM global o faltan intervalos.
            cortes_actuales.append(mejor_corte_iteracion)
            cortes_actuales.sort()
            global_caim = mejor_caim_iteracion
            k += 1
        else:
            break
    return sorted(cortes_actuales)

def procesar_mi_csv(ruta_archivo, nombre_columna_clase):
    print(f"Leyendo archivo: {ruta_archivo}")
    
    #1.Cargar datos
    try:
        df = pd.read_csv(ruta_archivo)
    except FileNotFoundError:
        print("ERROR!!! No se encuentra el archivo. Revisa la ruta.")
        return

    #2.Validar que existe la columna clase
    if nombre_columna_clase not in df.columns:
        print(f"ERROR!!! La columna '{nombre_columna_clase}' no existe en el CSV.")
        print(f"Columnas disponibles: {list(df.columns)}")
        return

    #Separar X (atributos) e y (clase)
    y = df[nombre_columna_clase]
    #Quitamos la clase para quedarnos solo con lo que vamos a discretizar
    df_features = df.drop(columns=[nombre_columna_clase])
    #Filtrar solo columnas numéricas (CAIM solo trabaja con continuos/numéricos)
    df_numerico = df_features.select_dtypes(include=[np.number])
    cols_descartadas = set(df_features.columns) - set(df_numerico.columns)
    if cols_descartadas:
        print(f"NOTA: Se ignorarán estas columnas no numéricas: {cols_descartadas}")
    diccionario_cortes = {}
    
    #3.Procesar
    print("\n--- Iniciando Algoritmo CAIM ---")
    for columna in df_numerico.columns:
        print(f"Procesando: {columna}...")
        try:
            cortes = obtener_cortes_atributo(df_numerico[columna], y)
            diccionario_cortes[columna] = cortes
            print(f"Cortes encontrados: {cortes}")
        except Exception as e:
            print(f"Error procesando {columna}: {e}")

    #4.Transformar y Guardar
    print("\nGenerando archivo discretizado")
    df_resultado = df.copy() # Copiamos el original para mantener columnas de texto.
    
    for columna, cortes in diccionario_cortes.items():
        bins = [-np.inf] + cortes + [np.inf]
        # Reemplazamos la columna numérica original por la discretizada.
        df_resultado[columna] = pd.cut(df_numerico[columna], bins=bins, labels=False, include_lowest=True)
    
    # Nombre del archivo de salida.
    nombre_salida = f"discretizado_{os.path.basename(ruta_archivo)}"
    df_resultado.to_csv(nombre_salida, index=False)
    
    print(f"ÉXITO!!! Archivo guardado como: {nombre_salida}")
    print(df_resultado.head())


if __name__ == "__main__":
    mi_archivo = "iris.csv" #Colocar el nombre de la base de datos.
    mi_columna_clase = "clase" #Colocar el nombre de la columna clase u objetivo.
    #Ejecutamos con el archivo y el target.

    procesar_mi_csv(mi_archivo, mi_columna_clase)
