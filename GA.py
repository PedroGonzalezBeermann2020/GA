import numpy as np
from sklearn.cross_decomposition import CCA
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import normalize

from docx import Document
from docx.shared import Inches
from datetime import datetime
import base64
from openpyxl import load_workbook

n_components=5
def generar_poblacion(tamaño_poblacion,num_variables):
    arr0 = np.zeros(num_variables,dtype=int)
    indices = np.random.choice(range(num_variables), size=5, replace=False)
    arr0[indices] = 1
    for i in range(tamaño_poblacion-1):
        arr = np.zeros(num_variables,dtype=int)
        indices = np.random.choice(range(num_variables), size=5, replace=False)
        arr[indices] = 1
        arr0 = np.vstack([arr0, arr])

    return arr0

def evaluar_aptitud(individuo,X,Y):
    
    
    X_seleccionadas=X.iloc[:,individuo==1]
    n_components=min(X_seleccionadas.shape[1], Y.shape[1])
    if n_components>5:
        n_components=5
    
    if X_seleccionadas.shape[1]==0:
        return 0
    cca=CCA(n_components=n_components)
    cca.fit(X_seleccionadas,Y)
    X_c,Y_c=cca.transform(X_seleccionadas,Y)
    c = np.array([np.corrcoef(X_c[:, i], Y_c[:, i])[0, 1] for i in range(n_components)])
    wilks_lambda=np.prod(1-c*c)
    return 1-wilks_lambda

def seleccion(poblacion, fitness):
    # Normalizamos los valores de fitness
    fitness = np.array(fitness)
    fitness_prob = fitness / fitness.sum()
    
    # Selección proporcional al fitness
    indices = np.random.choice(len(poblacion), size=len(poblacion), p=fitness_prob)
    return poblacion[indices]

def cruzar(padres, tasa_cruce=0.8):
    hijos = []
    for i in range(0, len(padres), 2):
        padre1, padre2 = padres[i], padres[i+1]
        if np.random.rand() < tasa_cruce:
            punto_cruce = np.random.randint(1, len(padre1)-1)
            hijo1 = np.concatenate([padre1[:punto_cruce], padre2[punto_cruce:]])
            hijo2 = np.concatenate([padre2[:punto_cruce], padre1[punto_cruce:]])
            hijos.append(hijo1)
            hijos.append(hijo2)
        else:
            hijos.append(padre1)
            hijos.append(padre2)
    return np.array(hijos)

def intercambiar(arreglo):

    # Paso 1: Encontrar las posiciones de los 1s y los 0s
    posiciones_1 = np.where(arreglo == 1)[0]
    posiciones_0 = np.where(arreglo == 0)[0]
    if len(posiciones_1) > 0 and len(posiciones_0) > 0:
        pos_1 = np.random.choice(posiciones_1)
        pos_0 = np.random.choice(posiciones_0)

        # Paso 3: Intercambiar el 1 y el 0 en las posiciones seleccionadas
        arreglo[pos_1], arreglo[pos_0] = arreglo[pos_0], arreglo[pos_1]

    return arreglo

def mutar(hijos, tasa_mutacion=0.01):
    for hijo in hijos:
        for i in range(len(hijo)):
            if np.random.rand() < tasa_mutacion:
                hijos=intercambiar(hijos) # Cambia de 1 a 0 o viceversa
                #hijo[i] = 1 - hijo[i]

    return hijos

def grafica(X_c,Y_c):
    plt.figure(figsize=(14, 8))
    for i in range(n_components):
        plt.subplot(2, 3, i+1)  # Assuming you have up to 6 canonical variates
        plt.scatter(X_c[:, i], Y_c[:, i], c='b', marker='o')
        plt.xlabel(f'Canonical Variate {i+1} (X)')
        plt.ylabel(f'Canonical Variate {i+1} (Y)')
        plt.title(f'Canonical Correlation {i+1}')
        plt.grid(True)

    plt.tight_layout()
    plt.savefig('grafica.png')
    plt.show()
    plt.close()

def biplot(Y,X,X_c, Y_c, labels_X, labels_Y,x_loads,y_loads):
    #plt.figure(figsize=(8, 6))
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    X_c[:, 0]=(X_c[:, 0]-np.mean(X_c[:, 0]))/np.std(X_c[:, 0])
    X_c[:, 1]=(X_c[:, 1]-np.mean(X_c[:, 1]))/np.std(X_c[:, 1])
    Y_c[:, 0]=(Y_c[:, 0]-np.mean(Y_c[:, 0]))/np.std(Y_c[:, 0])
    Y_c[:, 1]=(Y_c[:, 1]-np.mean(Y_c[:, 1]))/np.std(Y_c[:, 1])
    
    # Plot the canonical variables
    
    ss=np.sum(np.sqrt((X_c[:, 0]-Y_c[:, 0])**2+(X_c[:, 1]-Y_c[:, 1])**2))
    
    
    ax[0].scatter(Y_c[:, 0], Y_c[:, 1], color='blue', label='Y Canonical Variables',s=50)
    ax[0].scatter(X_c[:, 0], X_c[:, 1], color='red', label='X Canonical Variables',s=20)
    
    for i in range(X_c.shape[0]):
        ax[0].text(X_c[i, 0]*1.0+0.1, X_c[i, 1]*1.0, str(i+1), color='red')
    for i in range(Y_c.shape[0]):
        ax[0].text(Y_c[i, 0]*1.0+0.1, Y_c[i, 1]*1.0, str(i+1), color='blue')

    #scaler = StandardScaler()
    #x_loads= scaler.fit_transform(cca.x_loadings_)
    #y_loads= scaler.fit_transform(cca.y_loadings_)
    #x_loads=cca.x_loadings_
    #y_loads=cca.y_loadings_
    #x_loads = normalize(cca.x_loadings_, axis=0)
    #y_loads = normalize(cca.y_loadings_, axis=0)
    # Add vectors for X (arrows)
    #ax[0].set_xlim([-1, 1])
    #ax[0].set_ylim([-1, 1])
    
    for i in range(x_loads.shape[1]):
        ax[1].arrow(0, 0, x_loads[i, 0]*1.0, x_loads[i, 1]*1.0, color='red', alpha=0.5, head_width=0.05)
        ax[1].text(x_loads[i, 0]*1.1, x_loads[i,1]*1.1, labels_X[i], color='red')
    
    # Add vectors for Y (arrows)
    for i in range(Y.shape[1]):
        ax[1].arrow(0, 0, y_loads[i, 0]*1.0, y_loads[i, 1]*1.0, color='blue', alpha=0.5, head_width=0.05)
        ax[1].text(y_loads[i, 0]*1.1, y_loads[i,1]*1.1, labels_Y[i], color='blue')
    
    ax[0].set_xlabel('Canonical Component 1')
    ax[0].set_ylabel('Canonical Component 2')
    ax[0].set_title('Biplot of Canonical Correlation Analysis')
    ax[0].grid(True)
    
    ax[1].set_xlabel('Canonical Component 1')
    ax[1].set_ylabel('Canonical Component 2')
    ax[1].set_title('Biplot of Canonical Correlation Analysis')
    ax[1].grid(True)
    
    
    plt.tight_layout()  # Automatically adjust spacing
    plt.savefig('grafica2.png')
    plt.show()
    plt.close()
    return ss
	
def convertBase64(image_path):
    # Path to the image file

    # Open the image file in binary mode
    with open(image_path, 'rb') as image_file:
        # Read the image file as bytes
        image_data = image_file.read()

        # Encode the image data to Base64
        base64_encoded_data = base64.b64encode(image_data)

        # Convert Base64 bytes to string
        base64_image = base64_encoded_data.decode('utf-8')

    # Print the Base64 string
    return "<img src='data:image/png;base64,"+base64_image+"' />"
	

	