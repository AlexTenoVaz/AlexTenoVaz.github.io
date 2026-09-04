# Librerías básicas
from math import factorial
from itertools import product

# Librerías externas
import numpy as np
import networkx as nx
import random as rd

# Sympy
from sympy import (
    symbols, I, diff, simplify, nsimplify, oo, zoo,
    solve, Poly, SparseMatrix, expand, sympify, Pow, Add, Mul, Integer, Rational, sqrt
)

import itertools

from fractions import Fraction
import sympy as sp
from collections import defaultdict

from itertools import combinations

 
# Set of global variables
x, y, z, t = symbols('x y z t')


def custom_sort(vectors):
    """
    Orndena una lista de vectores basándose en suma y elementos.
    
    Esta función toma una lista de vectores (tuplas o listas) y los ordena
    de forma ascendente. El criterio de ordenación principal es la suma de
    los componentes. Si dos vectores tienen la misma suma, se utiliza el
    vector invertido como criterio de desempate (orden lexicográfico inverso).
    

    Parameters
    ----------
    vectors : list of tuples or list of lists
        Una lista que contiene los vectores numéricos a ordenar.
        Ejemplo: [(1, 3), (2, 2), (4, 0)]

    Returns
    -------
    list
        Una nueva lista con los vectores ordenados según el criterio definido.

    Examples
    --------
    >>> v = [(1, 4), (2, 3), (5, 0)] 
    >>> # Todos suman 5. Se desempata por el último elemento: 0 < 3 < 4
    >>> custom_sort(v)
    [(5, 0), (2, 3), (1, 4)]
    """
    return sorted(vectors, key=lambda v: (sum(v),) + v[::-1])


def default_col_vec(number_of_vars=3, jac_order=3):
    """
    Genera una lista ordenada de vectores de exponentes (multi-índices).

    Crea todas las combinaciones posibles de vectores de tamaño `number_of_vars`
    cuya suma de componentes esté entre 1 y `jac_order`. Los resultados se
    ordenan utilizando la lógica de `custom_sort`.

    Parameters
    ----------
    number_of_vars : int, optional
        El número de variables (dimensión del vector).
        Por defecto es 3.
    jac_order : int, optional
        El orden máximo (suma máxima de los componentes del vector).
        Por defecto es 3.

    Returns
    -------
    list of tuples
        Lista de tuplas ordenadas donde cada tupla representa un vector
        válido.

    Examples
    --------
    >>> # Con valores por defecto (vars=3, order=3)
    >>> default_col_vec()
    [(0, 0, 1), (0, 1, 0), (1, 0, 0), ... (1, 1, 1), ...]

    >>> # Solo vectores de dimensión 2 que sumen hasta 2
    >>> default_col_vec(number_of_vars=2, jac_order=2)
    [(0, 1), (1, 0), (0, 2), (1, 1), (2, 0)]
    """
    natural_numbers = list(range(jac_order + 1))
    return custom_sort([
        vec for vec in product(natural_numbers, repeat=number_of_vars)
        if 1 <= sum(vec) <= jac_order
    ])

def default_row_vec(number_of_vars=3, jac_order=4):
    """
    Genera una lista ordenada de vectores de exponentes (multi-índices).

    Crea todas las combinaciones posibles de vectores de tamaño `number_of_vars`
    cuya suma de componentes esté entre 0 y `jac_order`-1. Los resultados se
    ordenan utilizando la lógica de `custom_sort`.

    Parameters
    ----------
    number_of_vars : int, optional
        El número de variables (dimensión del vector).
        Por defecto es 3.
    jac_order : int, optional
        El orden máximo (suma máxima de los componentes del vector).
        Por defecto es 3.

    Returns
    -------
    list of tuples
        Lista de tuplas ordenadas donde cada tupla representa un vector
        válido.

    Examples
    --------
    >>> # Con valores por defecto (vars=3, order=3)
    >>> default_col_vec()
    [(0, 0, 0), (0, 1, 0), (1, 0, 0), ... (1, 1, 1), ...]

    >>> # Solo vectores de dimensión 2 que sumen hasta 2
    >>> default_col_vec(number_of_vars=2, jac_order=2)
    [(0,0), (0, 1), (1, 0)]
    """
    natural_numbers = list(range(jac_order + 1))
    return custom_sort([
        vec for vec in product(natural_numbers, repeat=number_of_vars)
        if 0 <= sum(vec) <= jac_order - 1
    ])


def derivative_dictionary(
        f, 
        fac = False, 
        col_vectors = None, 
        variables_list = None,  
        number_of_vars=3, 
        jac_order_local=4):
    """
    Calcula todas las derivadas parciales de un polinomio según multiíndices.

    Para cada vector en `col_vectors`, calcula la derivada parcial
    correspondiente.  
    Si `fac=True`, divide la derivada por el producto factorial
    de los órdenes, como en desarrollos de Taylor.

    Parámetros
    ----------
    f : sympy.Expr
        Polinomio o función simbólica a derivar.
    fac : bool, opcional
        Si True, divide por factoriales (coeficientes del Taylor).
    col_vectors : list of tuple, opcional
        Multiíndices de orden de derivación.
    variables_list : list, opcional
        Variables respecto a las cuales se derivará.
    number_of_vars : int, opcional
        Número de variables si no se pasa `variables_list`.
    jac_order_local : int, opcional
        Orden máximo usado si `col_vectors` es None.

    Retorna
    -------
    dict
        Diccionario {multiíndice : derivada no nula}.
    """
    if variables_list is None:
        # por defecto usamos los símbolos globales x,y,z (o menos si number_of_vars<3)
        base_vars = (x, y, z)
        variables_list = list(base_vars[:number_of_vars])

    if col_vectors is None:
        col_vectors = default_col_vec(number_of_vars=number_of_vars, jac_order=jac_order_local)
    
    if col_vectors is None:
        col_vectors = default_col_vec()
    derivative_dict = {}
    
    if fac == False:
        for vec in col_vectors:
            deriv = f
            for i, order in enumerate(vec):
                deriv = diff(deriv, variables_list[i], order)
            if deriv != 0:
                derivative_dict[vec] = deriv
    else:
        derivative_dict = {}
        for vec in col_vectors:
            deriv = f
            factorial_vector = 1
            for h in range(len(vec)):
                factorial_vector *= factorial(vec[h])
            for i, order in enumerate(vec):
                deriv = diff(deriv, variables_list[i], order)
            deriv /= factorial_vector
            if deriv != 0:
                derivative_dict[vec] = deriv
    return derivative_dict
    

# ===== MATRIZ JACOBIANA SIN DIVIDIR POR EL FACTORIAL ===== #


def pseudo_jacobian_matrix(f, row_vectors = None, col_vectors = None, variables_list = None, number_of_vars=3, jac_order_local=4):
    """
    Construye la matriz jacobiana ampliada sin dividir por factoriales.

    Usa los multiíndices fila y columna para determinar qué derivadas
    aparecen en cada entrada.  
    La entrada (i, j) es la derivada correspondiente al vector
    `col_vectors[j] - row_vectors[i]`, cuando todos los componentes son ≥ 0.

    Parámetros
    ----------
    f : sympy.Expr
        Polinomio base.
    row_vectors : list of tuple, opcional
        Multiíndices fila.
    col_vectors : list of tuple, opcional
        Multiíndices columna.
    variables_list : list, opcional
        Variables de derivación.
    number_of_vars : int
        Dimensión del sistema.
    jac_order_local : int
        Orden máximo.

    Retorna
    -------
    sympy.SparseMatrix
        Matriz jacobiana simbólica dispersa.
    """
    
    
    if row_vectors is None:
        row_vectors = default_row_vec(number_of_vars=number_of_vars, jac_order=jac_order_local)
    if col_vectors is None:
        col_vectors = default_col_vec(number_of_vars=number_of_vars, jac_order=jac_order_local)

    if variables_list is None:
        # por defecto usamos los símbolos globales x,y,z (o menos si number_of_vars<3)
        base_vars = (x, y, z)
        variables_list = list(base_vars[:number_of_vars])

    derivative_dict = derivative_dictionary(f, fac=False, col_vectors=col_vectors, variables_list=variables_list,
                                            number_of_vars=number_of_vars, jac_order_local=jac_order_local)

    
    derivative_dict = derivative_dictionary(f,False)
    num_rows = len(row_vectors)
    num_cols = len(col_vectors)
    
    sparse_entries = {}
    
    for r_idx, row in enumerate(row_vectors):
        for c_idx, col in enumerate(col_vectors):
            subtracted_vector = tuple(c - r for r, c in zip(row, col))
            
            if all(entry >= 0 for entry in subtracted_vector):
                matching_derivative = derivative_dict.get(subtracted_vector, 0)
                if matching_derivative != 0:
                    sparse_entries[(r_idx, c_idx)] = matching_derivative

    M = SparseMatrix(num_rows, num_cols, sparse_entries)
    return M


# ===== MATRIZ JACOBIANA ===== #
def jacobian_matrix(f, row_vectors = None, col_vectors = None,  variables_list=None, number_of_vars=3, jac_order_local=4):
    
    """
    Construye la matriz jacobiana ampliada agregando divisiones factoriales.

    Igual que `pseudo_jacobian_matrix`, pero cada derivada está normalizada
    dividiéndola por el producto de factoriales del multiíndice.

    Parámetros
    ----------
    f : sympy.Expr
        Polinomio base.
    row_vectors : list of tuple, opcional
    col_vectors : list of tuple, opcional
    variables_list : list, opcional
    number_of_vars : int
    jac_order_local : int

    Retorna
    -------
    sympy.SparseMatrix
        Matriz jacobiana simbólica normalizada.
    """
    
    if row_vectors is None:
        row_vectors = default_row_vec(number_of_vars=number_of_vars, jac_order=jac_order_local)
    if col_vectors is None:
        col_vectors = default_col_vec(number_of_vars=number_of_vars, jac_order=jac_order_local)
        
    if variables_list is None:
        # por defecto usamos los símbolos globales x,y,z (o menos si number_of_vars<3)
        base_vars = (x, y, z)
        variables_list = list(base_vars[:number_of_vars])

    derivative_dict = derivative_dictionary(f, fac=True, col_vectors=col_vectors, variables_list=variables_list,
                                            number_of_vars=number_of_vars, jac_order_local=jac_order_local)
    
    derivative_dict = derivative_dictionary(f, True)
    num_rows = len(row_vectors)
    num_cols = len(col_vectors)
    
    sparse_entries = {}
    
    for r_idx, row in enumerate(row_vectors):
        for c_idx, col in enumerate(col_vectors):
            subtracted_vector = tuple(c - r for r, c in zip(row, col))
            
            if all(entry >= 0 for entry in subtracted_vector):
                matching_derivative = derivative_dict.get(subtracted_vector, 0)
                if matching_derivative != 0:
                    sparse_entries[(r_idx, c_idx)] = matching_derivative

    M = SparseMatrix(num_rows, num_cols, sparse_entries)
    return M


# ===== ORDEN DE UN POLINOMIO ===== #
def order_pol(p):
    """
    Calcula el orden mínimo de t en un polinomio (con multiplicación auxiliar por t).

    Convierte la expresión en un polinomio en t, luego toma el exponente mínimo
    entre todos los monomios.

    Parámetros
    ----------
    p : sympy.Expr

    Retorna
    -------
    int
        Orden mínimo en t.
    """
    
    p = Poly(p* t**2, t)  # convertir a polinomio multiplicado por t
    if p.is_zero:
        return 0
    return min(p.monoms())[0]  # el menor exponente de t

# ===== ORDEN DE UN POLINOMIO CON EL FACTOR DE CORRECCIÓN t ===== #
def order_pol_nt(p):
    
    """
    Calcula el orden de t en un polinomio.

    Útil cuando el parámetro ya está incorporado explícitamente en la expresión.

    Parámetros
    ----------
    p : sympy.Expr

    Retorna
    -------
    int
        Orden mínimo en t.
    """
    p = Poly(p, t)  # convertir a polinomio
    if p.is_zero:
        return 0
    return min(p.monoms())[0]  # el menor exponente de t

# ===== CREA LA MATRIZ DE EXPONENTES ===== #
def order_matrix(f, orders_list, row_vectors= None, col_vectors = None, variables_list = [x,y,z], number_of_vars=3, jac_order_local=4):
    
    """
    Construye la matriz de órdenes evaluando la jacobiana en una curva dada.

    Sustituye x, y, z por potencias de t según `orders_list`,
    obtiene la matriz jacobiana simbólica y aplica `order_pol`
    a cada entrada.

    Parámetros
    ----------
    f : sympy.Expr
    orders_list : list of int
        Exponentes de t para cada variable.
    row_vectors, col_vectors : list of tuple
        Multiíndices.
    variables_list : list of sympy.Symbol
    number_of_vars : int
    jac_order_local : int

    Retorna
    -------
    sympy.Matrix
        Matriz cuyos coeficientes son órdenes de t.
    """
    
    if row_vectors is None:
        row_vectors = default_row_vec(number_of_vars=number_of_vars, jac_order=jac_order_local)
    if col_vectors is None:
        col_vectors = default_col_vec(number_of_vars=number_of_vars, jac_order=jac_order_local)
    
    
    subs_dict = {x:t**orders_list[0], y: t**orders_list[1], z: t**orders_list[2]}
    
    Mtx = pseudo_jacobian_matrix(f, row_vectors, col_vectors, None, number_of_vars, jac_order_local).subs(subs_dict)
    return Mtx.applyfunc(order_pol)


# ===== CREA LA GRAFICA DIRIGIDA ===== #
def build_flow_graph(M_array, exclude_cols= None):
    
    """
    Construye el grafo dirigido para resolver el problema de matching óptimo.

    Modela el problema como un flujo con costo mínimo:
    - nodos rᵢ representan filas,
    - nodos cⱼ representan columnas,
    - se conecta s → rᵢ → cⱼ → t.

    Parámetros
    ----------
    M_array : array-like
        Matriz (exponentes o pesos).
    exclude_cols : iterable, opcional
        Columnas que se excluyen del matching.

    Retorna
    -------
    networkx.DiGraph
        Grafo listo para pasarse a `network_simplex`.
    """
    
    if exclude_cols is None:
        exclude_cols = set()
    else:
        exclude_cols = set(exclude_cols)
    
    M_exp = np.array([[int(el) for el in row] for row in M_array.tolist()], dtype=int)
    rows, cols = M_exp.shape
    
    row_nodes = ["r{}".format(i) for i in range(rows)]
    
    source = "s"
    sink = "t"
    
    F = nx.DiGraph()
    F.add_node(source)
    F.add_node(sink)
    F.add_nodes_from(row_nodes)
    col_nodes_subset = ["c{}".format(j) for j in range(cols)]
    F.add_nodes_from(col_nodes_subset)

    # Demandas
    for n in row_nodes + col_nodes_subset:
        F.nodes[n]['demand'] = 0
        F.nodes[source]['demand'] = -rows
        F.nodes[sink]['demand'] = rows

    # Source -> filas
    for r in row_nodes:
        F.add_edge(source, r, capacity=1, weight=0)

    # Columnas -> sink
    for c in col_nodes_subset:
        F.add_edge(c, sink, capacity=1, weight=0)

    # Filas -> columnas con pesos, excluyendo columnas prohibidas
    for i in range(rows):
        for j in range(cols):
            if j in exclude_cols:
                continue
            if M_array[i, j] != 0:
                F.add_edge(f"r{i}", f"c{j}", capacity=1, weight=int(M_array[i, j]))
    return F


# ===== ENCONTRAR MATRIZ OPTIMA ===== #
def solve_matching(M_array, exclude_cols=None):
    """
    Resuelve el matching óptimo usando el método de flujo con costo mínimo.

    Usa `network_simplex` de networkx para obtener una asignación
    fila-columna de costo mínimo.

    Parámetros
    ----------
    M_array : array-like
        Matriz de pesos.
    exclude_cols : iterable, opcional
        Columnas prohibidas.

    Retorna
    -------
    matching_local : list of tuple
        Lista de pares (fila, columna) asignados.
    cost_local : int
        Costo total del matching.
    """
    
    if exclude_cols is None:
        exclude_cols = set()
    
    M_exp = np.array([[int(el) for el in row] for row in M_array.tolist()], dtype=int)
    rows, cols = M_exp.shape
    
    row_nodes = ["r{}".format(i) for i in range(rows)]
    
    F = build_flow_graph(M_array, exclude_cols)
    flowCost, flowDict = nx.network_simplex(F)
    matching_local = []
    for r in row_nodes:
        for c, fval in flowDict[r].items():
            if fval > 0:
                matching_local.append((r, c))
    cost_local = sum(int(M_array[int(r[1:]), int(c[1:])]) for r, c in matching_local)
    return matching_local, cost_local


# ===== ENCONTRAR SEGUNDA MATRIZ OPTIMA ===== #
def second_matching(M_array,cols_selected1, cost1):
    """
    Busca un segundo matching óptimo prohibiendo progresivamente columnas previas.

    Intenta excluir una columna del matching inicial y resolver nuevamente
    el problema.  
    Se acepta un matching solo si:
      - tiene tamaño completo,
      - tiene el mismo costo que el matching inicial.

    Parámetros
    ----------
    M_array : array-like
    cols_selected1 : list of int
        Columnas del primer matching óptimo.
    cost1 : int
        Costo del primer matching.

    Retorna
    -------
    segunda_submatriz : ndarray o None
    segunda_cols : list of int o None
    segunda_matching : list of tuple o None
    segunda_cost : int o None
    """
    
    M_exp = np.array([[int(el) for el in row] for row in M_array.tolist()], dtype=int)
    rows, cols = M_exp.shape
    
    segunda_submatriz = None
    segunda_cols = None
    segunda_matching = None
    segunda_cost = None

    # Probar columnas del primer matching de mayor a menor
    for col_to_exclude in reversed(cols_selected1):
        exclude_set = {col_to_exclude}

        try:
            m2, cost2 = solve_matching(M_exp, exclude_cols=exclude_set)

            # Solo aceptamos matching completo y mismo costo
            if len(m2) == rows and cost2 == cost1:
                # *** CORRECCIÓN: usar las columnas reales de m2 ***
                cols_selected2 = sorted(int(c[1:]) for _, c in m2)
                segunda_submatriz = M_exp[:, cols_selected2]
                segunda_cols = cols_selected2
                segunda_matching = m2
                segunda_cost = cost2
                break  # Detener búsqueda al encontrar otra óptima
        except nx.NetworkXUnfeasible:
            continue
    if segunda_submatriz is not None:
        return segunda_submatriz, segunda_cols, segunda_matching, segunda_cost
    else:
        return segunda_submatriz, segunda_cols, segunda_matching, segunda_cost
    


"""
Funciones para hallar curvas
"""

def racionaliza_lista(lista):
    """
    Convierte cada elemento de una lista en una expresión racional de SymPy.

    Parámetros
    ----------
    lista : list
        Lista de expresiones simbólicas.

    Retorna
    -------
    list
        Lista con expresiones racionalizadas mediante `nsimplify`.
    """
    return [nsimplify(expr, rational=True) for expr in lista]

def curve_on_ED(div, chart, pol_fijo = None, pol1 = None, pol2 = None, variables_list=None):
    """
    Construye dos curvas parametrizadas sobre un divisor excepcional.

    La función fija una variable (según `chart`), sustituye polinomios
    predefinidos en las otras y resuelve para obtener curvas que pasan
    por el divisor excepcional.

    Parámetros
    ----------
    div : sympy.Expr
        Ecuación del divisor excepcional.
    chart : int
        Índice de la variable usada como carta.
    pol_fijo : sympy.Expr, opcional
        Sustitución fija para la variable principal.
    pol1, pol2 : sympy.Expr, opcional
        Sustituciones alternativas para la otra variable.
    variables_list : list of sympy.Symbol, opcional

    Retorna
    -------
    (list, list)
        Dos curvas parametrizadas (C1, C2) como listas de expresiones.
    """
    
    if variables_list is None:
        var_list = [x, y, z]
    else:
        var_list = list(variables_list)
        
    var_list_nc = [var for i, var in enumerate(var_list) if i != chart]
    
    if pol_fijo is None: pol_fijo = t**2 # Un polinomio fijo para CARTA FIJA

    # Polinomios fijos para la variable CARTA NO FIJA
    if pol1 is None:  pol1 = I*t*sqrt(t + 1)
    if pol1 is None: pol2 = I*t*sqrt(t - 1)
    
    sols1 = (solve(div.subs({var_list[chart]:pol_fijo, var_list_nc[1]:pol1}), var_list_nc[0]))
    sols2 = (solve(div.subs({var_list[chart]:pol_fijo, var_list_nc[1]:pol2}), var_list_nc[0]))

    mapa_C1 = {
        var_list[chart]: pol_fijo,
        var_list_nc[1]: pol1,
        var_list_nc[0]: rd.choice(sols1)
    }
    
    mapa_C2 = {
        var_list[chart]: pol_fijo,
        var_list_nc[1]: pol2,
        var_list_nc[0]: rd.choice(sols2)
    }
    
    C1 = [mapa_C1[x], mapa_C1[y], mapa_C1[z]]
    C2 = [mapa_C2[x], mapa_C2[y], mapa_C2[z]]
    
    return C1, C2

def curva(cur, ordexpl):
    """
    Aplica una transformación multiplicativa iterada a una curva dada.

    Para cada índice en `ordexpl`, multiplica las otras coordenadas por
    la coordenada correspondiente, expandiendo y simplificando.

    Parámetros
    ----------
    cur : list of sympy.Expr
        Curva inicial.
    ordexpl : list of int
        Índices sobre los que se aplica la multiplicación.

    Retorna
    -------
    list of sympy.Expr
        Curva transformada y racionalizada.
    """
    Ncur = cur.copy()
    for idx in reversed(ordexpl):
        multiplicador = Ncur[idx]
        for i in range(len(Ncur)):
            if i != idx:
                Ncur[i] = Ncur[i] * multiplicador
                Ncur[i] = Ncur[i].expand()
                Ncur[i] = Ncur[i].simplify(complex=True)
    return Ncur


def orden_entero(expr, var=t):
    """
    Obtiene el orden entero de var en una expresión expresada como producto.

    Solo analiza expresiones tipo producto (Mul) o potencias `var**n`.

    Parámetros
    ----------
    expr : sympy.Expr
    var : sympy.Symbol

    Retorna
    -------
    int
        Exponente total de `var` en la expresión.
    """
    
    expr = expr.simplify()
    grado = 0
    
    if isinstance(expr, Mul):
        for arg in expr.args:
            # Potencia directa como t**n
            if isinstance(arg, Pow) and arg.base == var:
                grado += arg.exp
            # Caso t
            elif arg == var:
                grado += 1
        return int(grado)
    
    # Si es solo t**n
    if isinstance(expr, Pow) and expr.base == var:
        return int(expr.exp)
    if expr == var:
        return 1
    
    return 0


from sympy import expand, Add, Mul, Pow

def grado_maximo(expr, var=t):
    """
    Calcula el mayor exponente de `var` que aparece en la expresión.

    Parámetros
    ----------
    expr : sympy.Expr
    var : sympy.Symbol

    Retorna
    -------
    int
        Mayor exponente de `var`.
    """
    
    expr = expand(expr)
    
    # Si no depende de var
    if not expr.has(var):
        return 0
    
    # Si es suma → revisar cada término
    if isinstance(expr, Add):
        return max(grado_maximo(term, var) for term in expr.args)
    
    grado = 0
    
    # Producto
    if isinstance(expr, Mul):
        for arg in expr.args:
            if isinstance(arg, Pow) and arg.base == var:
                grado += arg.exp
            elif arg == var:
                grado += 1
            else:
                grado += grado_maximo(arg, var)
        return int(grado)
    
    # Potencia directa
    if isinstance(expr, Pow) and expr.base == var:
        return int(expr.exp)
    
    # Caso var
    if expr == var:
        return 1
    
    return 0

from sympy import Add, Mul, Pow, expand

def grado_parte_polinomial(expr, var = t):
    """
    Obtiene el grado en `var` considerando solo factores monomiales directos,
    ignorando radicales o funciones.

    Para sumas devuelve el máximo grado de los términos.
    """
    
    expr = expand(expr)
    
    # suma → máximo
    if isinstance(expr, Add):
        return min(grado_parte_polinomial(term, var) for term in expr.args)
    
    grado = 0
    
    # producto → sumar solo factores directos
    if isinstance(expr, Mul):
        for arg in expr.args:
            if isinstance(arg, Pow) and arg.base == var:
                grado += arg.exp
            elif arg == var:
                grado += 1
            # ignorar todo lo demás (sqrt, funciones, etc.)
        return int(grado)
    
    # potencia directa
    if isinstance(expr, Pow) and expr.base == var:
        return int(expr.exp)
    
    if expr == var:
        return 1
    
    return 0

def find_all_optimal_matchings(M_array, initial_cols, initial_cost, max_depth=None):
    """
    Busca exhaustivamente todos los matchings óptimos de igual costo.

    Genera diferentes prohibiciones de columnas del matching inicial
    y explora recursivamente nuevas soluciones viables.

    Parámetros
    ----------
    M_array : array-like
    initial_cols : list of int
        Columnas del matching inicial.
    initial_cost : int
        Costo del matching inicial.
    max_depth : int, opcional
        Profundidad máxima de búsqueda.

    Retorna
    -------
    list of tuple
        Lista de soluciones (columnas, matching, costo).
    """
    M_exp = np.array([[int(el) for el in row] for row in M_array.tolist()], dtype=int)
    rows, cols = M_exp.shape

    all_matchings = []
    visited_sets = set()  # Para evitar repetir combinaciones
    
    def cols_to_tuple(c):
        return tuple(sorted(c))
    
    # Añadimos el primero
    all_matchings.append((initial_cols, None, initial_cost))
    visited_sets.add(cols_to_tuple(initial_cols))
    
    queue = [(set(initial_cols), initial_cost)]
    depth = 1
    
    while queue:
        cols_set, cost_ref = queue.pop(0)
        if max_depth and depth > max_depth:
            break
        
        # Probar excluir cada subconjunto no vacío de las columnas ya encontradas
        for r in range(1, len(cols_set) + 1):
            for exclude_subset in itertools.combinations(cols_set, r):
                exclude_set = set(exclude_subset)
                try:
                    m2, cost2 = solve_matching(M_exp, exclude_cols=exclude_set)
                    if len(m2) == rows and cost2 == cost_ref:
                        cols_selected2 = sorted(int(c[1:]) for _, c in m2)
                        key = cols_to_tuple(cols_selected2)
                        if key not in visited_sets:
                            visited_sets.add(key)
                            all_matchings.append((cols_selected2, m2, cost2))
                            queue.append((set(cols_selected2), cost2))
                except nx.NetworkXUnfeasible:
                    continue
        
        depth += 1

    return all_matchings


def find_k_optimal_matchings(M_array, k=5):
    """
    Encuentra hasta k matchings óptimos distintos de igual costo mínimo.
    
    El algoritmo obtiene primero un matching óptimo base y luego genera
    nuevas soluciones variando prohibiciones de columnas: para cada
    matching previamente encontrado, se excluye temporalmente una de
    sus columnas y se intenta construir un nuevo matching de igual costo.
    Solo se aceptan soluciones completas y cuyo conjunto de columnas no
    coincida con ninguno anterior.
    
    Parámetros
    ----------
    M_array : array-like
        Matriz de costos sobre la cual se buscan los matchings.
    k : int, opcional
        Número máximo de matchings óptimos distintos que se desean obtener.
    
    Retorna
    -------
    list of dict
        Lista de soluciones encontradas. Cada elemento contiene:
        - 'index' : índice consecutivo de la solución.
        - 'cols'  : lista ordenada de columnas seleccionadas.
        - 'cost'  : costo óptimo asociado.
    """
    
    M_exp = np.array([[int(el) for el in row] for row in M_array.tolist()], dtype=int)
    rows, cols = M_exp.shape

    # Almacenaremos las soluciones
    matchings = []
    cols_selected_list = []
    costs = []

    # === Matching base ===
    m1, cost_opt = solve_matching(M_exp)
    if len(m1) < rows:
        raise ValueError("No se encontró un matching completo para la primera matriz.")
    cols1 = sorted(int(c[1:]) for _, c in m1)
    matchings.append(m1)
    cols_selected_list.append(cols1)
    costs.append(cost_opt)

    # === Buscamos otros matchings distintos ===
    intentos = 0
    while len(matchings) < k:
        intentos += 1
        found = False

        # Probamos excluir cada columna de cada matching anterior
        for prev_cols in cols_selected_list:
            for col_to_exclude in prev_cols:
                exclude_set = set()
                # Excluimos esa columna de todos los previos
                for all_prev in cols_selected_list:
                    exclude_set.update(all_prev)
                exclude_set.remove(col_to_exclude)

                try:
                    m_new, cost_new = solve_matching(M_exp, exclude_cols=exclude_set)

                    if len(m_new) == rows and cost_new == cost_opt:
                        cols_new = sorted(int(c[1:]) for _, c in m_new)
                        if cols_new not in cols_selected_list:
                            matchings.append(m_new)
                            cols_selected_list.append(cols_new)
                            costs.append(cost_new)
                            found = True
                            break
                except nx.NetworkXUnfeasible:
                    continue
            if found:
                break

        if not found:
            break  # ya no hay más matchings distintos posibles
        if intentos > 3 * k:
            break  # seguridad para evitar bucles infinitos

    # Empaquetamos resultados
    result = []
    for i, (m, cols, cost) in enumerate(zip(matchings, cols_selected_list, costs), 1):
        #subM = M_exp[:, cols]
        result.append({
            "index": i,
            # "matching": m,
            "cols": cols,
            "cost": cost,
            #"submatrix": subM
        })

    return result


def next_matching(M_array, prev_cols_list, cost_target):
    """
    Busca un único matching óptimo adicional que no repita ningún
    conjunto de columnas previamente encontrado.
    
    La función explora alternativas excluyendo, una por una, columnas
    pertenecientes a los matchings previos. Para cada exclusión, intenta
    reconstruir un matching completo con el mismo costo objetivo. Si se
    encuentra una solución cuyo conjunto de columnas no coincide con
    ninguno de los anteriores, se devuelve inmediatamente.
    
    Parámetros
    ----------
    M_array : array-like
        Matriz de costos sobre la cual se buscan matchings adicionales.
    prev_cols_list : list of list of int
        Conjuntos de columnas ya utilizados en matchings previos.
    cost_target : int
        Costo óptimo que deben cumplir las soluciones buscadas.
    
    Retorna
    -------
    tuple
        (submatriz, columnas, matching, costo) si encuentra una solución nueva,
        o (None, None, None, None) si no existe un matching óptimo distinto.
    """
    M_exp = np.array([[int(el) for el in row] for row in M_array.tolist()], dtype=int)
    rows, cols = M_exp.shape
    
    next_submatrix = None
    next_cols = None
    next_matching = None
    next_cost = None

    # conjunto total de columnas que aparecen en cualquier matching previo
    all_prev_cols = set(c for cols in prev_cols_list for c in cols)

    # probamos excluir distintas combinaciones (una columna de cada matching previo)
    for prev_cols in prev_cols_list:
        for col_to_exclude in reversed(prev_cols):
            exclude_set = {col_to_exclude}
            try:
                m_new, cost_new = solve_matching(M_exp, exclude_cols=exclude_set)
                
                if len(m_new) == rows and cost_new == cost_target:
                    cols_new = sorted(int(c[1:]) for _, c in m_new)
                    
                    # ✅ checamos que no coincida con ningún matching previo
                    if cols_new not in prev_cols_list:
                        next_submatrix = M_exp[:, cols_new]
                        next_cols = cols_new
                        next_matching = m_new
                        next_cost = cost_new
                        return next_submatrix, next_cols, next_matching, next_cost

            except nx.NetworkXUnfeasible:
                continue

    # Si no encontramos ninguno distinto
    return None, None, None, None


def unique_lists(lista):
    vistos = set()
    resultado = []
    for sub in lista:
        t = tuple(sub)
        if t not in vistos:
            vistos.add(t)
            resultado.append(sub)
    return resultado



def random_det(A, vars, trials=3):
    """Regresa un hash basado en determinantes evaluados aleatoriamente"""
    evals = []
    for _ in range(trials):
        subs = {v: rd.randint(1, 5) for v in vars}
        val = A.subs(subs).det()
        evals.append(val)
    return tuple(evals)

def unique_by_determinant(M, cols_list, vars, trials=3):
    seen = {}
    result = []

    for cols in cols_list:
        # construir la submatriz
        sub = M[:, cols]

        # hash probabilístico del determinante
        h = random_det(sub, vars, trials)

        if h not in seen:
            seen[h] = cols
            result.append(cols)

    return result

def gauss_custom(M):
    M = M.copy()
    rows, cols = M.shape
    
    pivot_rows = set()
    pivot_cols = set()
    
    # Recorremos las filas
    for i in range(rows):
        # Busca el primer 1 en la fila i
        ones_positions = [j for j in range(cols) if M[i, j] == 1]
        if not ones_positions:
            continue  # si no hay 1, saltamos
        pivot_col = ones_positions[0]
        pivot_rows.add(i)
        pivot_cols.add(pivot_col)
        
        # Anular otras filas en la misma columna
        for k in range(rows):
            if k != i and M[k, pivot_col] != 0:
                factor = M[k, pivot_col]
                M[k, :] = M[k, :] - factor * M[i, :]
    
    # Reducción por columnas (opcional)
    for j in range(cols):
        ones_positions = [i for i in range(rows) if M[i, j] == 1]
        if not ones_positions:
            continue
        pivot_row = ones_positions[0]
        pivot_rows.add(pivot_row)
        pivot_cols.add(j)
        for k in range(cols):
            if k != j and M[pivot_row, k] != 0:
                factor = M[pivot_row, k]
                M[:, k] = M[:, k] - factor * M[:, j]
    
    # Eliminar filas y columnas que contenían 1s
    remaining_rows = [i for i in range(rows) if i not in pivot_rows]
    remaining_cols = [j for j in range(cols) if j not in pivot_cols]
    M_reduced = M.extract(remaining_rows, remaining_cols)
    
    return M_reduced

# --- UTIL: get leading power and leading coefficient of expr around t=0 ---
def leading_term(expr, t, order_cut=5):
    """
    Return (alpha, r) such that expr ~ alpha * t^r (1 + O(t))
    Works for algebraic expressions composed of powers, polynomials and sqrt.
    Tries to expand expr as series up to small order_cut and finds minimal power.
    """
    ser = sp.series(expr, t, 0, order_cut).removeO()
    # ser is polynomial in t with possibly fractional powers if used with .expand?
    # Convert to terms and find min exponent
    terms = sp.Add.make_args(sp.simplify(ser))
    min_power = None
    coeff = 0
    for term in terms:
        # term = c * t**p  (maybe p integer). If nested, try to extract power
        term = sp.simplify(term)
        # get power: write term as a*t**p
        a, b = sp.fraction(sp.together(term))
        # fallback: use series of term
        # Use sp.order or as_coeff_exponent
        try:
            c, p = sp.sympify(term).as_coeff_exponent(t)
        except Exception:
            # numeric fallback: expand small series
            s2 = sp.series(term, t, 0, order_cut).removeO()
            c, p = s2.as_coeff_exponent(t)
        if min_power is None or p < min_power:
            min_power = p
            coeff = c
    return coeff, min_power

# --- UTIL: compute (r_x, alpha_x) for each coordinate robustly ---
def coord_lead(expr, t, series_order=8):
    # expand sqrt etc. as series to series_order
    ser = sp.series(expr, t, 0, series_order).removeO()
    # find lowest exponent in series
    # convert to sum of terms
    terms = sp.Add.make_args(sp.simplify(ser))
    powers = []
    for term in terms:
        c, p = term.as_coeff_exponent(t)
        powers.append((int(p) if p.is_Integer else p, c))
    # choose minimal p
    powers_sorted = sorted(powers, key=lambda pc: sp.Rational(pc[0]))
    # return first coefficient and exponent (as Rational)
    p0, c0 = powers_sorted[0]
    return sp.simplify(c0), sp.Rational(p0)


def leading_of_polynomial_monomial_list(monomials, alpha, r):
    """
    monomials: list of (coeff, (ax,by,cz))
    alpha: (alpha_x, alpha_y, alpha_z)
    r: (rx, ry, rz) rationals
    returns (v_min, coeff_sum) where v_min rational and coeff_sum symbolic
    """
    groups = defaultdict(list)
    ax = alpha; rx = r
    for coeff, (a,b,c) in monomials:
        # order
        v = sp.Rational(a)*rx[0] + sp.Rational(b)*rx[1] + sp.Rational(c)*rx[2]
        # effective coefficient
        C = simplify(Rational(coeff) * (ax[0]**a) * (ax[1]**b) * (ax[2]**c))
        groups[v].append(C)
    # find minimal v
    vmin = min(groups.keys())
    Csum = sum(groups[vmin])
    if Csum == 0:
        # cancellation: remove and retry (for production, loop)
        sorted_vs = sorted(groups.keys())
        for v in sorted_vs:
            s = sum(groups[v])
            if s != 0:
                return v, simplify(s)
        return None, 0
    return vmin, simplify(Csum)
# Nota: para tu caso real conviene construir monomials de los polinomios grandes
# automáticamente con sp.Poly pero con cuidado: Poly puede expandir todo; mejor
# construir la lista a partir del texto original o de la descomposición de términos.


def es_finito(expr):
    if expr in (oo, -oo, zoo, I*oo, -I*oo):
        return False
    return expr.is_finite is not False


def all_minors_near_optimal(M_orders, jac_mat, delta=2):
    """
    Encuentra todos los menores cuyo orden real es mínimo,
    permitiendo tolerancia en el costo del matching.

    Parámetros
    ----------
    M_orders : matriz de órdenes
    jac_mat : matriz jacobiana simbólica
    delta : tolerancia sobre costo mínimo

    Retorna
    -------
    lista de columnas óptimas
    lista de determinantes simbólicos
    orden mínimo real
    """

    rows, cols = M_orders.shape

    # ===== costo mínimo por matching =====
    _, cost_min = solve_matching(M_orders)

    candidates = []

    # ===== explorar todas combinaciones de columnas =====
    for cols_subset in combinations(range(cols), rows):

        # costo estimado = suma mínimos por fila restringido
        cost_est = sum(min(M_orders[i, j] for j in cols_subset) for i in range(rows))

        if cost_est <= cost_min + delta:
            candidates.append(cols_subset)

    print(f"Candidatos explorados: {len(candidates)}")

    # ===== calcular orden real =====
    dets = []
    orders_real = []

    for cols_subset in candidates:
        Det = jac_mat[:, cols_subset].det()
        ord_real = order_pol(Det)
        dets.append((cols_subset, Det))
        orders_real.append(ord_real)

    ord_min_real = min(orders_real)

    # ===== filtrar =====
    final = [cols for (cols, det), ordv in zip(dets, orders_real) if ordv == ord_min_real]
    final_dets = [det for (cols, det), ordv in zip(dets, orders_real) if ordv == ord_min_real]

    return final, final_dets, ord_min_real

import networkx as nx

def all_optimal_matchings(M_array):
    """
    Enumera todos los matchings óptimos quitando aristas,
    no columnas completas.
    """

    M_exp = np.array(M_array, dtype=int)
    rows, cols = M_exp.shape

    # matching base
    base_matching, cost_min = solve_matching(M_exp)

    solutions = []
    stack = [base_matching]

    seen = set()

    while stack:

        current = stack.pop()
        key = tuple(sorted(current))

        if key in seen:
            continue

        seen.add(key)
        solutions.append(current)

        # explorar quitando una arista
        for r, c in current:

            F = build_flow_graph(M_exp)

            if F.has_edge(r, c):
                F.remove_edge(r, c)

            try:
                cost, flowDict = nx.network_simplex(F)

                if cost == cost_min:

                    new_match = []
                    for rr in [f"r{i}" for i in range(rows)]:
                        for cc, fval in flowDict[rr].items():
                            if fval > 0:
                                new_match.append((rr, cc))

                    if len(new_match) == rows:
                        stack.append(new_match)

            except nx.NetworkXUnfeasible:
                pass

    # convertir a columnas
    cols_list = [sorted(int(c[1:]) for _, c in m) for m in solutions]

    # quitar duplicados
    unique_cols = []
    for c in cols_list:
        if c not in unique_cols:
            unique_cols.append(c)

    return unique_cols, cost_min

from itertools import combinations

def all_minors_by_estimated_order(M_orders, jac_mat, tolerance=3):

    rows, cols = M_orders.shape

    _, cost_min = solve_matching(M_orders)

    candidates = []

    for cols_subset in combinations(range(cols), rows):

        # estimación usando matching restringido
        subM = M_orders[:, cols_subset]

        try:
            _, cost_est = solve_matching(subM)
        except:
            continue

        if cost_est <= cost_min + tolerance:
            candidates.append(cols_subset)

    print("Candidatos:", len(candidates))

    # calcular orden real
    dets = []
    orders = []

    for cols_subset in candidates:
        Det = jac_mat[:, cols_subset].det()
        ord_real = order_pol(Det)
        dets.append((cols_subset, Det))
        orders.append(ord_real)

    ord_min_real = min(orders)

    final_cols = []
    final_dets = []

    for (cols_subset, Det), ordv in zip(dets, orders):
        if ordv == ord_min_real:
            final_cols.append(cols_subset)
            final_dets.append(Det)

    return final_cols, final_dets, ord_min_real

from itertools import combinations

def brute_minors_with_bound(M_orders, jac_mat):

    rows, cols = M_orders.shape

    # orden mínimo estimado global
    _, cost_min = solve_matching(M_orders)

    best_order = None
    best = []

    for cols_subset in combinations(range(cols), rows):

        # cota inferior
        lower_bound = sum(min(M_orders[i, j] for j in cols_subset) for i in range(rows))

        if lower_bound > cost_min + 5:
            continue

        Det = jac_mat[:, cols_subset].det()
        ord_real = order_pol(Det)

        if best_order is None or ord_real < best_order:
            best_order = ord_real
            best = [cols_subset]

        elif ord_real == best_order:
            best.append(cols_subset)

    return best, best_order

def find_all_optimal_matchings_cols(M_array, initial_matching, initial_cost, max_depth=None):

    M_exp = np.array([[int(el) for el in row] for row in M_array.tolist()], dtype=int)
    rows, cols = M_exp.shape

    def matching_to_cols(m):
        return tuple(sorted(int(c[1:]) for _, c in m))

    visited = set()
    results = []

    # inicial
    init_cols = matching_to_cols(initial_matching)
    visited.add(init_cols)
    results.append(list(init_cols))

    queue = [(initial_matching, initial_cost)]
    depth = 1

    while queue:
        matching, cost_ref = queue.pop(0)

        if max_depth and depth > max_depth:
            break

        cols_set = set(int(c[1:]) for _, c in matching)

        # probar excluir subconjuntos
        for r in range(1, len(cols_set)+1):
            for exclude_subset in itertools.combinations(cols_set, r):

                try:
                    m2, cost2 = solve_matching(M_exp, exclude_cols=set(exclude_subset))

                    if len(m2) == rows and cost2 == cost_ref:

                        cols_tuple = matching_to_cols(m2)

                        if cols_tuple not in visited:
                            visited.add(cols_tuple)
                            results.append(list(cols_tuple))
                            queue.append((m2, cost2))

                except nx.NetworkXUnfeasible:
                    continue

        depth += 1

    return results