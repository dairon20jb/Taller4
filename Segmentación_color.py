import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture as GMM
from sklearn.cluster import KMeans
from sklearn.utils import shuffle
from time import time
def recreate_image(centers, labels, rows, cols):
    d = centers.shape[1]
    image_clusters = np.zeros((rows, cols, d))
    label_idx = 0
    for i in range(rows):
        for j in range(cols):
            image_clusters[i][j] = centers[labels[label_idx]]
            label_idx += 1
    return image_clusters
if __name__ == '__main__':

    path = 'C:\PRUEBA\proc ima'
    image_name = 'bandera.png'
    path_file = os.path.join(path, image_name)
    image = cv2.imread(path_file)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    i = 1
    distancia = []
    method = ['kmeans', 'gmm']
    select = 1

    # Convert to floats instead of the default 8 bits integer coding. Dividing by
    # 255 is important so that plt.imshow behaves works well on float data (need to
    # be in the range [0-1])


    image = np.array(image, dtype=np.float64) / 255

    # Load Image and transform to a 2D numpy array.
    rows, cols, ch = image.shape
    assert ch == 3
    image_array = np.reshape(image, (rows * cols, ch))
    image_array_sample = shuffle(image_array, random_state=0)[:10000]
    n_colors = 0
    for i in range(10):
        n_colors = n_colors+1
        print("Fitting model on a small sub-sample of the data")
        t0 = time()

        if method[select] == 'gmm':
            model = GMM(n_components=n_colors).fit(image_array_sample)
        else:
            model = KMeans(n_clusters=n_colors, random_state=0).fit(image_array_sample)
        print("done in %0.3fs." % (time() - t0))

        # Get labels for all points
        print("Predicting color indices on the full image (GMM)")
        t0 = time()
        if method[select] == 'gmm':
            labels = model.predict(image_array)
            centers = model.means_
        else:
            labels = model.predict(image_array)
            centers = model.cluster_centers_
        print("done in %0.3fs." % (time() - t0))

        # Display all results, alongside original image
        suma_distancias = 0
        suma_distancias_d = 0
        for idx in range(len(image_array)):
            suma_distancias_d = abs(centers[labels[idx]]-image_array[idx])
            suma_distancias =  np.linalg.norm(suma_distancias_d) + suma_distancias
        #distancia.append((suma_distancias[0]+suma_distancias[1]+suma_distancias[2]))
        distancia.append(suma_distancias)
        # plt.figure(1)
        # plt.clf()
        # plt.axis('off'
        # plt.title('Original image')
        # plt.imshow(image)
        # plt.figure(1)
        # plt.clf()
        # plt.axis('off')
        # plt.title('Quantized image ({} colors, method={})'.format(n_colors, method[select]))
        # plt.imshow(recreate_image(centers, labels, rows, cols))
        # plt.show()
    plt.figure(2)
    plt.plot([1,2,3,4,5,6,7,8,9,10],(distancia))
    plt.show()


