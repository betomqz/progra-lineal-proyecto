# -*- coding: utf-8 -*-
"""
Created on Mon Nov 21 22:25:36 2022

@author: santi
"""
import numpy as np

def matriz_t0(A, b, c, M):
    '''
    Método para construir la Tabla 0 con los datos del problema.
    '''
    mat1 = []
    
    m = len(A)
    
    ceros = [0 for i in range(m)]
    
    for i,(coef, b0) in enumerate(zip(A, b)):
        aux = ceros.copy()
        aux[i] = 1
        
        coef = coef + aux + [b0]
        mat1 = mat1 + [coef]
        
    c = c + [M for i in range(m)] + [0]
    mat1 = mat1 + [c]
    
    canonicos = [len(A[0]) + i for i in range(m)]
    
    return mat1, canonicos

def reglaDeBland(tablaSimplex):
    '''
    Queremos la columna más a la izquierda con cr < 0.
    Si hay empate en el criterio de la variable de salida,
    elejimos la más arriba

    Args:
        tablaSimplex (TYPE): DESCRIPTION.

    Returns:
        None.

    '''
    m = len(A)
    
    cr = tablaSimplex[-1][:-1] #costos relativos
    
    if any(z < 0 for z in cr):
        colSal = -1
        for i, x in enumerate(cr):
            if x<0:
                colSal=i
            break
    else:
        return -1, tablaSimplex[-1][-1]
    
    bk = [a[-1] for a in tablaSimplex]
    yk = [a[colSal] for a in tablaSimplex]
    
    by = [b/y for b,y in zip(bk,yk) if y>0]
    
    p = min(by)
    
    renglonSal = by.index(p)
    
    return renglonSal, colSal

def pivoteo(tablaSimplex):
    
    renglonSal, colSal = reglaDeBland(tablaSimplex)
    
    if renglonSal == -1:
        return renglonSal, colSal
    
    tabla1 = []
    
    m = len(tablaSimplex)
    
    for i in range(m):
        tabla1 = tabla1 + []

    valorPivoteo = tablaSimplex[renglonSal][colSal]
    tabla1[renglonSal] = np.array(tablaSimplex[renglonSal]) / valorPivoteo

    for i, eq in enumerate(tablaSimplex):
        if i != renglonSal:
            mult = np.array(tabla1[renglonSal]) * tablaSimplex[colSal]
            tabla1[i] = np.array(tablaSimplex[i]) - mult

    return tabla1,1
    
def solver(A,b,c,M=100):
    
    t0,c =  matriz_t0(A, b, c,M)
    
    t1,z = pivoteo(t0)
    
    if t1 == -1:
        return (-1)*z
    else:
        while z != -1:
            t1, z = pivoteo(t1)
        return z
        
    


#%% Pruebas
c = [1, 1, 0, 0]

A = [[3, 4, 1, 0],
    [ 2,-1, 0,-1]]

b = [20,2]

z,t1 = solver(A,b,c)
