import numpy as np

def matriz_t0(A, b, c, M):
    '''
    Método para obtener la tabla cero del problema usando el método
    de la Gran M.

    EJEMPLO DE USO:
    >>> A = np.array([[3, 4, 1, 0],
                     [ 2,-1, 0,-1]])
    >>> b = np.array([20,2])
    >>> c = np.array([1, 1, 0, 0])
    >>> M = 100
    >>> matriz_t0(A, b, c, M)
    array([[  3.,   4.,   1.,   0.,   1.,   0.,  20.],
           [  2.,  -1.,   0.,  -1.,   0.,   1.,   2.],                                                                             [  1.,   1.,   0.,   0., 100., 100.,   0.]])  
    '''
    m = len(A)
    canon = np.eye(m, dtype=int)                     # Matriz identidad mxm
    mat1 = np.concatenate((A,canon), axis=1)         # Pegamos A con la identidad
    cr = np.append(c,[M]*m)                          # Vector de costos relativos
    mat1 = np.concatenate((mat1, np.array([cr])))    # Pegamos la matriz con los costos relativos
    b_ext = np.array([np.append(b, 0)])              # Construcción del vector b
    
    # Regresamos la matriz extendida con b
    return np.concatenate((mat1, b_ext.T), axis=1).astype('float64')

def reglaDeBland(tablaSimplex):
    '''
    Queremos la columna más a la izquierda con cr < 0.
    Si hay empate en el criterio de la variable de salida,
    elegimos la más arriba
    '''
    cr = tablaSimplex[-1][:-1] # costos relativos
    busq = np.where(cr < 0)[0]
    
    if len(busq) == 0: # No encontró; fin del problema
        return -1, -1
    else:
        colSal = busq[0]
        
    bk = tablaSimplex[:, -1]
    yk = tablaSimplex[:, colSal]
    
    by = np.empty(0)
    for b,y in zip(bk,yk):
        if y > 0:
            by = np.append(by, b/y)
        else:
            by = np.append(by, -1)
    
    valid = np.where(by >= 0)[0]
    
    if len(valid) == 0: # Todas las variables son menores que cero
        return -1, -2
    
    renglonSal = valid[by[valid].argmin()]
    
    return renglonSal, colSal

def pivoteo(tablaSimplex):
    '''
    Dada una tabla Simplex, este método pivotea sobre el elemento 
    que dictamina la regla de Bland y regresa el resultado.
    '''
    renglonSal, colSal = reglaDeBland(tablaSimplex)
    
    if colSal < 0: # Condiciones para detenerse
        return tablaSimplex, colSal
    
    m = len(tablaSimplex)
    
    valorPivoteo = tablaSimplex[renglonSal][colSal]
    tablaSimplex[renglonSal] = tablaSimplex[renglonSal] / valorPivoteo
    
    for i in range(m):
        if i != renglonSal and tablaSimplex[i][colSal] != 0:
            tablaSimplex[i] -= tablaSimplex[i][colSal] * tablaSimplex[renglonSal]
    
    return tablaSimplex, 1

def solver(A,b,c,M=100):
    '''
    Método para resolver un PPL planteado en su forma estándar. Utiliza
    el método de la Gran M y Simplex. Regresa la tabla en su forma final
    y el resultado de la función objetivo.
    '''
    t0 =  matriz_t0(A, b, c,M)
    
    for i in range(len(A)):
        t0[-1] += t0[i]*(-M)
    
    t1, z = pivoteo(t0) # Aquí z es la columna de salida y la usamos como control para saber si terminó.
    
    while z >= 0:
        t1, z = pivoteo(t1)
    
    if z == -2: # no está acotado
        print("PROBLEMA NO ACOTADO; última versión de la tabla:")
        return t1, np.nan
    
    return t1, (-1)*t1[-1][-1]

# Main
if __name__ == "__main__":
    print("----- PROBLEMA 0 -----")
    c = np.array([1, 1, 0, 0])
    A = np.array([[3, 4, 1, 0],
                [ 2,-1, 0,-1]])
    b = np.array([20,2])
    M = 100

    t1, z = solver(A, b, c, M)
    print(t1)
    print(f"\nEl valor   de la función objetivo es: {z}")

    print("\n\n----- PROBLEMA 1 -----")
    c1 = np.array([0, -9, -1, 0, 2, 1])
    A1 = np.array([[0, 5, 50, 1, 1, 0],
                [1, -15, 2, 0, 0, 0],
                [0, 1, 1, 0, 1, 1]])
    b1 = np.array([10, 2, 6])
    t1, z1 = solver(A1, b1, c1, M)
    print(t1)
    print(f"\nEl valor de la función objetivo es: {z1}")

    print("\n\n----- PROBLEMA 2 -----")
    c2 = np.array([-3, 1, 0, 0])
    A2 = np.array([[-1, 1, 1, 0],
                [2, 2, 0, -1]])
    b2 = np.array([5, 4])
    t2, z2 = solver(A2, b2, c2, M)
    print(t2)
    print(f"\nEl valor de la función objetivo es: {z2}")

    print("\n\n----- PROBLEMA 3 -----")
    c3 = np.array([-40, -30, 0, 0])
    A3 = np.array([[1, 1, 1, 0],
                [2, 1, 0, 1]])
    b3 = np.array([12, 16])
    t3, z3 = solver(A3, b3, c3, M)
    print(t3)
    print(f"\nEl valor de la función objetivo es: {z3}")

    print(print("\n\n----- PROBLEMAS PROFE -----"))
    c = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,-1,-1,0,0,0,0,0,0,0])
    b = np.array([2,2,0,0,0,2])
    A = np.array([
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 0, 0, 0, 0,-1, 0, 0, 0, 0],
        [0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 1,-1, 0,-1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,-1, 1, 0, 0,-1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1,-1, 0, 0, 0,-1, 0],
        [2, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,-1, 0, 0, 0, 0, 1]])
    t, z = solver(A, b, c, M)
    print(t)
    print(f"\nEl valor de la función objetivo es: {z}")