import cv2
import os
import numpy as np
import math
click = 0
refPt = list()
def click_event(event, x, y, flags, pafram):
    global click
    if click < 3 :
        if event == cv2.EVENT_LBUTTONDOWN:
            click = click + 1
            print(x, ",", y)
            refPt.append([x, y])
            print(len(refPt))
    else:
        click = 0
        cv2.destroyAllWindows()
if __name__ == '__main__':
    path1 = 'C:\PRUEBA'
    name1 = 'lena(1).png'
    path_file1 = os.path.join(path1, name1)
    I1 = cv2.imread(path_file1, 1)
    path2 = 'C:\PRUEBA'
    name2 = 'lena_warped.png'
    path_file2 = os.path.join(path2, name2)
    I2 = cv2.imread(path_file2, 1)
    cont = 0
    if cont == 0 :
        cv2.imshow("I1", I1)
        cv2.setMouseCallback("I1", click_event)
        cont=cont+1
        cv2.waitKey(0)
    if cont == 1 :
        cv2.imshow("I2", I2)
        cv2.setMouseCallback("I2", click_event)
        cv2.waitKey(0)
    pts1 = np.float32(refPt[0:3])
    pts2 = np.float32(refPt[3:6])

    M_affine = cv2.getAffineTransform(pts1, pts2)
    image_affine = cv2.warpAffine(I1, M_affine, I1.shape[:2])
    s0 = np.sqrt(((M_affine[0,0])**2) +((M_affine[1,0])**2))
    s1 = np.sqrt(((M_affine[0,1])**2) +((M_affine[1,1])**2))
    theta =np.arctan((M_affine[1,0]) / (M_affine[0,0]))
    #print(pts2)
    pruebasencilla= np.sin(90)
    graditos = pruebasencilla*180/np.pi
    print(pruebasencilla)
    print(pts1.shape)
    theta_grad= theta*180 / np.pi
    print(theta_grad,"ve que si")
    x0= (((M_affine[0,2])*np.cos(theta_grad))-((M_affine[1,2])*np.sin(theta_grad)))/s0
    x1= (((M_affine[0,2])*np.cos(theta_grad))-((M_affine[1,2])*np.sin(theta_grad)))/s1
    M_sim = np.float32([[s0 * np.cos(theta), -np.sin(theta), x0],[np.sin(theta), s1 * np.cos(theta), x1]])
    image_similarity = cv2.warpAffine(I1, M_sim, I1.shape[:2])
    vnorm = np.append(pts1.transpose(),np.array([[1,1,1]]), axis = 0)
    similitud_puntos= M_sim.dot(vnorm)
    Trans_similitud_puntos= similitud_puntos[:-1,:].transpose()
    error=np.linalg.norm(Trans_similitud_puntos-pts2,axis=1)
    print("La norma del error es :", error)
    cv2.imshow("Similar", image_similarity)
    cv2.imwrite(os.path.join(path1, 'Similar.png'), image_similarity)
    cv2.imshow("Affine", image_affine)
    cv2.imwrite(os.path.join(path1, 'Affine.png'), image_affine)
    cv2.imshow("Image2", I2)
    cv2.imwrite(os.path.join(path1, 'Image2.png'), I2)
    cv2.waitKey(0)