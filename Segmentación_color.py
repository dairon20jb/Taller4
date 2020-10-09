import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture as GMM
from sklearn.cluster import KMeans
from sklearn.utils import shuffle
from time import time
def recreate_image(centers, labels, rows, cols): # Construcción de imagen
    d = centers.shape[1]     #Tamaño de centers
    image_clusters = np.zeros((rows, cols, d))
    label_idx = 0
    for i in range(rows): # Construcción pixel a pixel
        for j in range(cols):
            image_clusters[i][j] = centers[labels[label_idx]]
            label_idx += 1
    return image_clusters  #  Retorna la imagen
if __name__ == '__main__':
    print("Digite la ruta de la imagen:")
    path =input()#'C:\PRUEBA\proc ima'
    print("Digite el nombre de la imagen:")
    image_name =input()#'bandera.png'
    path_file = os.path.join(path, image_name)
    image = cv2.imread(path_file)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Conversión de BGR a RGB
    i = 1
    distancia = [] # Creación de arreglo distancias
    print("Digite el método deseado (kmeans o gmm):")
    method = input()
    image = np.array(image, dtype=np.float64) / 255
    rows, cols, ch = image.shape # Imagen cargada
    assert ch == 3
    image_array = np.reshape(image, (rows * cols, ch)) # Arregló de la imagen
    image_array_sample = shuffle(image_array, random_state=0)[:10000]
    n_colors = 0
    for i in range(10):  # Se procede a aplicar el método para n_color desde 1 hasta 10
        n_colors = n_colors+1
        print("Fitting model on a small sub-sample of the data")
        t0 = time()  #Mide el tiempo de ejecución
        if method == 'gmm': #Se realiza el metodo gmm
            model = GMM(n_components=n_colors).fit(image_array_sample)
            print("gmm pred")
        else:     # Si el string recibido no es 'gmm' se procede con el método KMeans
            model = KMeans(n_clusters=n_colors, random_state=0).fit(image_array_sample)
            print("kmeans pred")
        print("done in %0.3fs." % (time() - t0))
        t0 = time()
        if method == 'gmm':
            labels = model.predict(image_array)
            centers = model.means_ #Valores de los centros
        else:
            labels = model.predict(image_array)
            centers = model.cluster_centers_ #Valores de los centros
        print("done in %0.3fs." % (time() - t0))
        # Display all results, alongside original image
        suma_distancias = 0
        suma_distancias_d = 0
        for idx in range(len(image_array)): # Suma de distancias intra_cluster
            suma_distancias_d = abs(centers[labels[idx]]-image_array[idx]) # Se realiza la suma píxel a píxel con los centros generados en cada color
            suma_distancias =  np.linalg.norm(suma_distancias_d) + suma_distancias
        distancia.append(suma_distancias) # En esta lista se guarda la suma de distancias para cada n_color
        plt.figure(1)  # Muestra la imagen original
        plt.clf()
        plt.axis('off'
        plt.title('Original image')
        plt.imshow(image)
        plt.figure(2)  # Muestra la imagen segmentada
        plt.clf()
        plt.axis('off')
        plt.title('Quantized image ({} colors, method={})'.format(n_colors, method[select]))
        plt.imshow(recreate_image(centers, labels, rows, cols))
        plt.show()
    plt.figure(3)
    plt.xlabel('n_color')
    plt.ylabel('Distancia intra cluster') # Se grafica la suma de distancias dependiendo del método seleccionado
    if method == "gmm" :
        plt.title('Suma de distancias intra cluster vs n_color para gmm:',)
    else :
        plt.title('Suma de distancias intra cluster vs n_color para k-means:', )
    plt.plot([1,2,3,4,5,6,7,8,9,10],(distancia))
    plt.show()


