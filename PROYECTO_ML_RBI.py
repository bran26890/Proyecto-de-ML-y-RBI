"""
=============================================================================
MAESTRÍA EN INTELIGENCIA ARTIFICIAL
ASIGNATURA: PROYECTO INTEGRADO (APRENDIZAJE AUTOMÁTICO + RAZONAMIENTO BAJO INCERTIDUMBRE)

TÍTULO: BENCHMARK DE ALGORITMOS ESTRUCTURALES: HC vs SA vs HÍBRIDO
AUTORES:
    Cruz Sánchez Erick Axel,
    Del Toro González Ronaldo Angelinni 
    Herrera Aldana Brandon Jhair

DESCRIPCIÓN:
Este script compara tres estrategias de aprendizaje de estructura en Redes Bayesianas:
  1. Hill Climbing (HC): Búsqueda Voraz Local.
  2. Simulated Annealing (SA): Búsqueda Estocástica Global.
  3. Híbrido (SA + HC): Exploración global seguida de refinamiento local.

METODOLOGÍA:
- Función de Score: BIC (Bayesian Information Criterion).
- Validación: K-Fold Cross Validation (k=5).
- Métricas: Precisión (Accuracy) en Test, Tiempo de Cómputo y Score BIC Final.
=============================================================================
"""

import pandas as pd #Manejo de datos.
import numpy as np #Operaciones matemáticas con los datos.
import networkx as nx #Manejo de grafos dirigidos.
import matplotlib.pyplot as plt #Imprimir visualmente las redes.
import random #Para números aleatorios.
import math #Librería matemática.
import time #Medir el tiempo de cada algoritmo.
import logging #Entorno para redes bayesianas.
from pgmpy.readwrite import XMLBIFWriter


#Configuración del entorno.
logger = logging.getLogger('pgmpy')
logger.setLevel(logging.ERROR) 

from sklearn.model_selection import KFold

#Importación robusta de librerías dadas las versiones y sea compatible.
try:
    from pgmpy.models import DiscreteBayesianNetwork as ModeloBayesiano
except ImportError:
    try:
        from pgmpy.models import BayesianModel as ModeloBayesiano
    except ImportError:
        from pgmpy.models import BayesianNetwork as ModeloBayesiano

from pgmpy.estimators import BayesianEstimator  
from pgmpy.inference import VariableElimination

# =============================================================================
# 1. FUNCIÓN DE EVALUACIÓN (SCORE BIC)
# =============================================================================
def calcular_bic_nodo(df, nodo, padres):
    """Calcula el componente local del BIC penalizando la complejidad."""
    N = len(df)
    if not padres: #Cálculo del BIC para un nodo sin padres.
        conteos = df[nodo].value_counts() #Conteo de frecuencias.
        log_likelihood = np.sum(conteos * np.log(conteos / N + 1e-10)) #Log-verosimilitud para ese nodo.
        num_params = len(conteos) - 1 #Número de parámetros.
    else: #Cálculo del BIC para un nodo con padres.
        cols = padres + [nodo] #Obtener las columnas de los padres y el nodo.
        conteos_conjuntos = df.groupby(cols).size() #Conteo de frecuencias.
        conteos_padres = df.groupby(padres).size() #Frecuencias d elos padres.
        log_likelihood = 0 #Inicializar la log-verosimilitud.
        #Obtener la log-verosimilitud tomando en cuenta a los padres.
        for index, count in conteos_conjuntos.items(): 
            val_padres = index[:-1] if len(padres) > 1 else index[0]
            count_parent = conteos_padres.get(val_padres, 1)
            prob = count / count_parent
            log_likelihood += count * np.log(prob + 1e-10)
        r_i = df[nodo].nunique()
        q_i = 1
        for p in padres: q_i *= df[p].nunique()
        num_params = q_i * (r_i - 1)
    return log_likelihood - (0.5 * np.log(N) * num_params) #Retornar la métrica completa BIC.

def obtener_score_total(grafo, df): #Obtener el BIC de una red bayesiana completa.
    """Suma scores locales (Descomponibilidad)."""
    score_total = 0
    for nodo in grafo.nodes(): #Por cada nodo del grafo.
        padres = list(grafo.predecessors(nodo)) #Obtener los padres.
        score_total += calcular_bic_nodo(df, nodo, padres) #Calular su score y realizar sumatoria.
    return score_total #Se retorna el score de una red completa.

# =============================================================================
# 2. OPERADORES DE NAVEGACIÓN
# =============================================================================
def aplicar_operacion(grafo, operacion): #Operadores para los arcos de las redes bayesianas.
    nuevo_grafo = grafo.copy() #El nuevo grafo será la copia del actual.
    tipo, u, v = operacion #Variables operables.
    if tipo == 'add': nuevo_grafo.add_edge(u, v) #Operación suma.
    elif tipo == 'del': nuevo_grafo.remove_edge(u, v) #Operación de eliminación.
    elif tipo == 'rev': #Operación invertir.
        nuevo_grafo.remove_edge(u, v) #Remover el arco actual.
        nuevo_grafo.add_edge(v, u) #Agregar el arco alrevez.
    return nuevo_grafo #Regresar el nuevo grafo válido a probar.

def generar_vecino_aleatorio(grafo, nodos): #Para generar el nuevo grafo a probar.
    for _ in range(50): #En un máximo de 50 iteraciones.
        u, v = random.sample(nodos, 2) #Elegir el nodo aleatoriamente.
        opciones = [] #Verificar que grafo es válido.
        if grafo.has_edge(u, v): opciones = [('del', u, v), ('rev', u, v)]
        elif not grafo.has_edge(v, u): opciones = [('add', u, v)]
        #Si es un nodo válido.
        if opciones:
            operacion = random.choice(opciones) #Realizar una operación aleatoria.
            temp_g = aplicar_operacion(grafo, operacion)
            if nx.is_directed_acyclic_graph(temp_g): #Necesario siempre asegurarse de que sea acíclico.
                return operacion, temp_g #Regresar la operación y el nodo válido.
    return None, grafo #De no aver operaciones válidas regresar el último al que se llego.

# =============================================================================
# 3. MÓDULO DE PREDICCIÓN (CON SUAVIZADO BAYESIANO)
# =============================================================================
def evaluar_capacidad_predictiva(grafo, df_train, df_test, target_col, state_names_global):
    """   Entrena usando Estimación Bayesiana (BDeu) para evitar probabilidades cero.    """
    if len(grafo.edges()) == 0:
        return 0.0 #Si el grafo no tiene arcos su precisión es 0%.
    try:
        modelo = ModeloBayesiano(list(grafo.edges())) #Crear el modelos.
        modelo.add_nodes_from(df_train.columns) #Agregar sus nodos.
        modelo.fit(df_train, #Entrenar el modelo usando BDeu.
                   estimator=BayesianEstimator, 
                   prior_type='BDeu', 
                   equivalent_sample_size=10, 
                   state_names=state_names_global)
        inferencia = VariableElimination(modelo) #Aplicar inferencia bayesiana.
        aciertos = 0 
        nodos_conocidos = set(modelo.nodes())
        
        for _, fila in df_test.iterrows(): #por cada fila de la base de datos.
            try:
                #Recolectar los target de la base de datos.
                real = fila[target_col]
                #Recolectar la evidencia de todos los nodos.
                evidencia = fila.drop(labels=[target_col]).to_dict()
                evidencia = {k: v for k, v in evidencia.items() if k in nodos_conocidos}
                #Realizar la predicción.
                pred = inferencia.map_query([target_col], evidence=evidencia, show_progress=False)
                #Hacer conteo de todos los aciertos.
                if pred[target_col] == real:
                    aciertos += 1
            except:
                continue 
        #Calcular la precisión y devolverla.
        return aciertos / len(df_test) if len(df_test) > 0 else 0.0
    except Exception:
        return 0.0

# =============================================================================
# 4. ALGORITMOS DE BÚSQUEDA
# =============================================================================
def algoritmo_hill_climbing(df, grafo_inicial=None, max_iter=50):
    """
    Hill Climbing modificado para aceptar un punto de partida (grafo_inicial).
    Esto permite usarlo como fase de refinamiento en el algoritmo híbrido.
    """
    #Crear el grafo inicial vacío.
    columnas = list(df.columns)
    if grafo_inicial is None:
        grafo = nx.DiGraph()
        grafo.add_nodes_from(columnas)
    else:
        grafo = grafo_inicial.copy()
    #Calcular el score de este grafo vacío.
    score_actual = obtener_score_total(grafo, df)
    for _ in range(max_iter):
        #Buscar algún vecino con mejor score. 
        mejor_vecino_score = score_actual
        mejor_vecino_grafo = None
        intentos = 0
        while intentos < 30: # Muestreo estocástico para velocidad
            #Realizar operaciones para explorar el espacio.
            u, v = random.sample(columnas, 2)
            op = 'add'
            if grafo.has_edge(u,v): op = random.choice(['del', 'rev'])
            cand = aplicar_operacion(grafo, (op, u, v))
            if nx.is_directed_acyclic_graph(cand):
                s = obtener_score_total(cand, df)
                if s > mejor_vecino_score:
                    mejor_vecino_score = s
                    mejor_vecino_grafo = cand
            intentos += 1
        #Quedarse con el mejor grafo en score en cada iteración.
        if mejor_vecino_score > score_actual:
            score_actual = mejor_vecino_score
            grafo = mejor_vecino_grafo
        else:
            break #Convergencia local
            
    return grafo

def algoritmo_simulated_annealing(df, max_iter=200, temp_inicial=100):
    """Simulated Annealing estándar para exploración global."""
    columnas = list(df.columns)
    grafo = nx.DiGraph()
    grafo.add_nodes_from(columnas)
    score_actual = obtener_score_total(grafo, df)
    #El mejor grafo actual es el vacío inicialmente.
    mejor_global = grafo.copy()
    mejor_score_global = score_actual
    T = temp_inicial
    #Realizar la búsqueda en un máximo de 200 iteraciones.
    for _ in range(max_iter):
        #Obtener el vecino de manera aleatoria.
        op, vecino_g = generar_vecino_aleatorio(grafo, columnas)
        if op is None: continue 
        #BIC del vcino obtenido.
        score_vecino = obtener_score_total(vecino_g, df)
        #Calcular cuanta diferencia hubo.
        delta = score_vecino - score_actual
        #Obtener probabilidad si no hay división por cero.
        try: prob = math.exp(delta / T)
        except: prob = 0
        #Si hay diferencia positiva nos quedamos con ese 
        # mejor score y vecino.
        if delta > 0 or random.random() < prob:
            grafo = vecino_g
            score_actual = score_vecino
            if score_actual > mejor_score_global:
                mejor_score_global = score_actual
                mejor_global = grafo.copy()
        #Reducir la temperatura.
        T *= 0.95
        if T < 0.1: break #Criterio de paro por baja temperatura.
            
    return mejor_global

def algoritmo_hibrido(df):
    """
    Estrategia Híbrida:
    1. Fase Exploratoria: Simulated Annealing busca una buena zona global.
    2. Fase de Explotación: Hill Climbing refina esa solución para llegar a la cima local.
    """
    #Fase 1:Buscar la mejor zona con recocido simulado
    grafo_sa = algoritmo_simulated_annealing(df, max_iter=150, temp_inicial=100)
    #Fase 2:Pasamos el grafo de SA como inicio para refinar con Hill Climbing
    grafo_final = algoritmo_hill_climbing(df, grafo_inicial=grafo_sa, max_iter=30)
    return grafo_final #Retornar el grafo final.

# =============================================================================
# 5. VISUALIZACIÓN COMPARATIVA (TRIPLE)
# =============================================================================
def visualizar_comparacion_triple(res_hc, res_sa, res_hib, nombre):
    """Genera un panel con las 3 redes finales."""
    plt.figure(figsize=(18, 6))
    
    # --- HILL CLIMBING ---
    plt.subplot(1, 3, 1)
    try: pos1 = nx.shell_layout(res_hc['grafo'])
    except: pos1 = nx.circular_layout(res_hc['grafo'])
    nx.draw(res_hc['grafo'], pos1, with_labels=True, node_color='#a8dadc', node_size=1200, font_weight='bold')
    plt.title(f"Hill Climbing\nAcc: {res_hc['acc']:.1%} | BIC: {res_hc['bic']:.0f}\nTiempo: {res_hc['time']:.2f}s", fontsize=10)

    # --- SIMULATED ANNEALING ---
    plt.subplot(1, 3, 2)
    try: pos2 = nx.shell_layout(res_sa['grafo'])
    except: pos2 = nx.circular_layout(res_sa['grafo'])
    nx.draw(res_sa['grafo'], pos2, with_labels=True, node_color='#ffbe0b', node_size=1200, font_weight='bold')
    plt.title(f"Simulated Annealing\nAcc: {res_sa['acc']:.1%} | BIC: {res_sa['bic']:.0f}\nTiempo: {res_sa['time']:.2f}s", fontsize=10)

    # --- HÍBRIDO ---
    plt.subplot(1, 3, 3)
    try: pos3 = nx.shell_layout(res_hib['grafo'])
    except: pos3 = nx.circular_layout(res_hib['grafo'])
    nx.draw(res_hib['grafo'], pos3, with_labels=True, node_color='#ff006e', node_size=1200, font_weight='bold')
    plt.title(f"Híbrido (SA+HC)\nAcc: {res_hib['acc']:.1%} | BIC: {res_hib['bic']:.0f}\nTiempo: {res_hib['time']:.2f}s", fontsize=10)

    plt.suptitle(f"Comparativa Final - Dataset: {nombre}", fontsize=16)
    plt.tight_layout()
    plt.show()


# =============================================================================
# 6. GUARDAR LA RED EN XML
# =============================================================================
def guardar_red_bayesiana_xml(grafo, df, nombre_archivo, state_names_global):
    """ Guarda la red bayesiana aprendida en formato XMLBIF estándar para Weka"""
    try:
        #1.Reconstruir el modelo con la estructura aprendida.
        modelo = ModeloBayesiano(list(grafo.edges()))
        modelo.add_nodes_from(df.columns)
        #2.Entrenar los parámetros (CPDs) asegurando que coincidan con los estados globales.
        modelo.fit(
            df,
            estimator=BayesianEstimator,
            prior_type='BDeu',
            equivalent_sample_size=10,
            state_names=state_names_global
        )
        #3.Escribir usando XMLBIFWriter
        writer = XMLBIFWriter(modelo)
        writer.write_xmlbif(filename=nombre_archivo)  # <--- ESTE ES EL CAMBIO CLAVE
        print(f"Red guardada correctamente en formato XML:{nombre_archivo}")

    except Exception as e:
        print(f"ERROR!!! Falló al guardar XML:{e}")

# =============================================================================
# 7. EJECUCIÓN PRINCIPAL
# =============================================================================
mis_archivos = ["glass_discretizado.csv"] #Introducir el nombre de tu archivo.
TARGET_COL = 'target' #Introducir el nombre de la columna objetivo o clase.
K_FOLDS = 5 #Elegir el número de folds para la validación cruzada.

print("\n" + "="*80)
print("  MAESTRÍA EN IA - PROYECTO FINAL")
print(f"  BENCHMARK: HC vs SA vs HÍBRIDO (k={K_FOLDS})")
print("="*80)

for arch in mis_archivos:
    try:
        print(f"\nProcesando Base de Datos: [{arch}]")
        print("-" * 60)
        #Verificar que no halla columnas vacías.
        df = pd.read_csv(arch)
        if 'Unnamed: 0' in df.columns: df = df.drop(columns=['Unnamed: 0'])
        df = df.astype(str) 
        #Verifica el nombre de la columnas no tenga caracteres incorrectos.
        STATE_NAMES_GLOBAL = {col: list(df[col].unique()) for col in df.columns}
        #Asegurarse de que exista la columna objetivo.
        if TARGET_COL not in df.columns: continue
        #Particionar los datos.
        kf = KFold(n_splits=K_FOLDS, shuffle=True, random_state=42)
        # Almacenes de métricas
        metrics = {
            'HC': {'acc': [], 'time': [], 'bic': []},
            'SA': {'acc': [], 'time': [], 'bic': []},
            'HIB': {'acc': [], 'time': [], 'bic': []}
        }
        last_results = {} # Para guardar grafos del último fold
        fold = 1
        #En cada fold realizar lo siguiente..
        for train_idx, test_idx in kf.split(df):
            train_data = df.iloc[train_idx]
            test_data = df.iloc[test_idx]
            print(f"   > Fold {fold}/{K_FOLDS}...", end=" ")
            #1.Búqueda con HILL CLIMBING
            start = time.time()
            g_hc = algoritmo_hill_climbing(train_data)
            t_hc = time.time() - start
            acc_hc = evaluar_capacidad_predictiva(g_hc, train_data, test_data, TARGET_COL, STATE_NAMES_GLOBAL)
            bic_hc = obtener_score_total(g_hc, train_data)
            
            metrics['HC']['acc'].append(acc_hc); metrics['HC']['time'].append(t_hc); metrics['HC']['bic'].append(bic_hc)
            last_results['HC'] = {'grafo': g_hc, 'acc': acc_hc, 'bic': bic_hc, 'time': t_hc}

            #2.SIMULATED ANNEALING
            start = time.time()
            g_sa = algoritmo_simulated_annealing(train_data)
            t_sa = time.time() - start
            acc_sa = evaluar_capacidad_predictiva(g_sa, train_data, test_data, TARGET_COL, STATE_NAMES_GLOBAL)
            bic_sa = obtener_score_total(g_sa, train_data)
            
            metrics['SA']['acc'].append(acc_sa); metrics['SA']['time'].append(t_sa); metrics['SA']['bic'].append(bic_sa)
            last_results['SA'] = {'grafo': g_sa, 'acc': acc_sa, 'bic': bic_sa, 'time': t_sa}

            #3.HÍBRIDO (SA + HC)
            start = time.time()
            g_hib = algoritmo_hibrido(train_data)
            t_hib = time.time() - start
            acc_hib = evaluar_capacidad_predictiva(g_hib, train_data, test_data, TARGET_COL, STATE_NAMES_GLOBAL)
            bic_hib = obtener_score_total(g_hib, train_data)
            
            metrics['HIB']['acc'].append(acc_hib); metrics['HIB']['time'].append(t_hib); metrics['HIB']['bic'].append(bic_hib)
            last_results['HIB'] = {'grafo': g_hib, 'acc': acc_hib, 'bic': bic_hib, 'time': t_hib}
            
            print(f"| Acc -> HC:{acc_hc:.0%} SA:{acc_sa:.0%} HIB:{acc_hib:.0%}")
            fold += 1
            
        #Reporte final de resultados.
        print("-" * 80)
        print(f"{'ALGORITMO':<15} | {'ACCURACY (Prom)':<15} | {'TIEMPO (Prom)':<15} | {'BIC (Prom)':<15}")
        print("-" * 80)
        
        m_hc_acc = np.mean(metrics['HC']['acc'])
        m_sa_acc = np.mean(metrics['SA']['acc'])
        m_hib_acc = np.mean(metrics['HIB']['acc'])
        
        print(f"{'Hill Climbing':<15} | {m_hc_acc:<15.2%} | {np.mean(metrics['HC']['time']):<15.4f} | {np.mean(metrics['HC']['bic']):<15.1f}")
        print(f"{'Sim. Annealing':<15} | {m_sa_acc:<15.2%} | {np.mean(metrics['SA']['time']):<15.4f} | {np.mean(metrics['SA']['bic']):<15.1f}")
        print(f"{'Híbrido':<15} | {m_hib_acc:<15.2%} | {np.mean(metrics['HIB']['time']):<15.4f} | {np.mean(metrics['HIB']['bic']):<15.1f}")
        print("="*80)
        
        visualizar_comparacion_triple(last_results['HC'], last_results['SA'], last_results['HIB'], arch)
        guardar_red_bayesiana_xml(last_results['HIB']['grafo'],df,f"red_HIB.xml", STATE_NAMES_GLOBAL)
        guardar_red_bayesiana_xml(last_results['HC']['grafo'],df,f"red_HC.xml", STATE_NAMES_GLOBAL)
        guardar_red_bayesiana_xml(last_results['SA']['grafo'],df,f"red_SA.xml", STATE_NAMES_GLOBAL)

    except FileNotFoundError:
        print(f"ERROR!!! Archivo {arch} no encontrado.")