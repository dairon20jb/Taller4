
import cv2
import os
import numpy as np
click = 0
refPt = list()
def click_event(event, x, y, flags, pafram):  # Por cada click se guardan las coordenadas en las que se posicione el mouse
    global click
    if click < 3 :  # Si click es mayor a tres se cierra la imagen
        if event == cv2.EVENT_LBUTTONDOWN:
            click = click + 1
            print(x, ",", y)   # Se imprimen las coordenadas guardadas
            refPt.append([x, y])
            print(len(refPt))
    else:
        click = 0
        cv2.destroyAllWindows()
if __name__ == '__main__':
    print("Digite la ruta de las imagenes:")  # Se digita la ruta, el nombre de la imagen I1 y el nombre de la imagen I2
    path1 = input()#'C:\PRUEBA'
    print("Digite nombre de la imagen I1:")
    name1 = input()#'lena(1).png'
    path_file1 = os.path.join(path1, name1)
    I1 = cv2.imread(path_file1, 1)
    print("Digite nombre de la imagen I2:")
    name2 = input()#'lena_warped.png'
    path_file2 = os.path.join(path1, name2)
    I2 = cv2.imread(path_file2, 1)
    cont = 0
    if cont == 0 :  # Mientras click sea menor a 3 se tomaran puntos de la primera imagen
        cv2.imshow("I1", I1)
        cv2.setMouseCallback("I1", click_event)
        cont=cont+1
        cv2.waitKey(0)
    if cont == 1 :  # Se toman los tres puntos de la segunda imagen
        cv2.imshow("I2", I2)
        cv2.setMouseCallback("I2", click_event)
        cv2.waitKey(0)
    pts1 = np.float32(refPt[0:3]) # Seleccionando puntos de la primera imagen
    pts2 = np.float32(refPt[3:6]) # Seleccionando puntos de la segunda imagen
    M_affine = cv2.getAffineTransform(pts1, pts2)  # Transformada Afín a partir de los puntos
    image_affine = cv2.warpAffine(I1, M_affine, I1.shape[:2]) # Aplicando la transformada Afín sobre la imagen
    # Parámetros para calcula la transformada de similitud
    s0 = np.sqrt(((M_affine[0,0])**2) +((M_affine[1,0])**2)) # Escalamiento en x
    s1 = np.sqrt(((M_affine[0,1])**2) +((M_affine[1,1])**2)) # Escalamiento en y
    theta =np.arctan((M_affine[1,0]) / (M_affine[0,0]))  # Rotación
    theta_grad= theta*180 / np.pi # Rotación en grados
    x0= (((M_affine[0,2])*np.cos(theta_grad))-((M_affine[1,2])*np.sin(theta_grad)))/s0 # Traslación en x
    x1= (((M_affine[0,2])*np.cos(theta_grad))-((M_affine[1,2])*np.sin(theta_grad)))/s1 # Traslación en y
    M_sim = np.float32([[s0 * np.cos(theta), -np.sin(theta), x0],[np.sin(theta), s1 * np.cos(theta), x1]]) # Matriz de similitud
    image_similarity = cv2.warpAffine(I1, M_sim, I1.shape[:2]) # Aplicando transformada de similitud en la imagen
    # Error
    # Para aplicar la transformada sobre los puntos se necesitan los mismos tamaños en los arreglos y disposición
    vnorm = np.append(pts1.transpose(),np.array([[1,1,1]]), axis = 0)
    similitud_puntos= M_sim.dot(vnorm)  # Transformada de similitud sobre los puntos
    Trans_similitud_puntos= similitud_puntos[:-1,:].transpose()  # Se necesita pts2 de 2x3 y trans_similitud de puntos
    error=np.linalg.norm(Trans_similitud_puntos-pts2,axis=1)  # Norma del error respecto a los puntos anotados de la imagen I2
    print("La norma del error es :", error)
    cv2.imshow("Similar", image_similarity)
    cv2.imwrite(os.path.join(path1, 'Similar.png'), image_similarity)
    cv2.imshow("Affine", image_affine)
    cv2.imwrite(os.path.join(path1, 'Affine.png'), image_affine)
    cv2.imshow("Image2", I2)
    cv2.imwrite(os.path.join(path1, 'Image2.png'), I2)
    cv2.waitKey(0)