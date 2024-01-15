# import numpy as np
# import cv2
# import matplotlib.pyplot as plt
# from PIL import Image

# crop_img = cv2.imread('C:/Users/usuario/Desktop/recortadas/energy_meter_10.jpg')

# # Verificar si un filtro se puede aplicar
# mask = np.zeros(crop_img.shape, dtype=np.uint8)
# gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
# thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

# ret,thresh1 = cv2.threshold(gray,150,300,cv2.THRESH_BINARY)
# ret,thresh2 = cv2.threshold(gray,100,200,cv2.THRESH_BINARY_INV)
# ret,thresh3 = cv2.threshold(gray,500,0,cv2.THRESH_TRUNC)
# ret,thresh4 = cv2.threshold(gray,150,200,cv2.THRESH_TOZERO)
# ret,thresh5 = cv2.threshold(gray,127,255,cv2.THRESH_TOZERO_INV)
# th2 = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
#             cv2.THRESH_BINARY,11,2)
# th3 = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
#             cv2.THRESH_BINARY,11,2)


# titles = ['Original Image','BINARY','BINARY_INV','TRUNC','TOZERO','TOZERO_INV', 'p1', 'p2']
# images = [crop_img, thresh1, thresh2, thresh3, thresh4, thresh5, th2, th3]
# for i in range(8):
#     plt.subplot(4,4,i+1),plt.imshow(images[i],'gray',vmin=0,vmax=255)
#     plt.title(titles[i])
#     plt.xticks([]),plt.yticks([])
# plt.show()

import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image

crop_img1 = cv2.imread('C:/Users/amnv2/Desktop/prueba/energy_meter_10.jpg')
crop_img2 = cv2.imread('C:/Users/amnv2/Desktop/prueba/energy_meter_14.jpg')
crop_img3 = cv2.imread('C:/Users/amnv2/Desktop/prueba/energy_meter_17.jpg')
crop_img4 = cv2.imread('C:/Users/amnv2/Desktop/prueba/energy_meter_5.jpg')
crop_img5 = cv2.imread('C:/Users/amnv2/Desktop/prueba/energy_meter_1.jpg')

i = [crop_img1, crop_img2, crop_img3, crop_img4, crop_img5]
x=0

for img in i:
    # Verificar si un filtro se puede aplicar
    if img is not None:
        x=x+1
        # Crear una máscara de ceros para cada imagen
        mask = np.zeros(img.shape, dtype=np.uint8)

        # Convertir la imagen a escala de grises
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Aplicar el umbral
        ret, thresh3 = cv2.threshold(gray, 500, 255, cv2.THRESH_TRUNC)

        #thresh3 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)

        # Otsu's thresholding after Gaussian filtering
        #blur = cv2.GaussianBlur(gray,(5,5),0)
        #ret3,thresh3 = cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

        cv2.imwrite("prueba3"+str(x)+".jpg", thresh3)
        print("imagen "+ "prueba"+str(x)+".jpg almacenada")

# import cv2
# import numpy as np

# def aplicar_umbralizacion(imagen):
#     gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
#     _, umbralizada = cv2.threshold(gris, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
#     return umbralizada

# def aplicar_filtrado_morfo(imagen):
#     suavizada = cv2.GaussianBlur(imagen, (5, 5), 0)
#     kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
#     morfo = cv2.morphologyEx(suavizada, cv2.MORPH_CLOSE, kernel)
#     return morfo

# def recortar_caracteres(imagen):
#     contornos, _ = cv2.findContours(imagen, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     caracteres = []

#     for contorno in contornos:
#         (x, y, w, h) = cv2.boundingRect(contorno)
#         caracter = imagen[y:y + h, x:x + w]
#         caracteres.append(caracter)

#     return caracteres

# def escalar_normalizar(caracter, ancho_deseado, alto_deseado):
#     caracter = cv2.resize(caracter, (ancho_deseado, alto_deseado))
#     caracter = caracter / 255.0  # Normalizar los valores de píxeles
#     return caracter

# def invertir_colores(imagen):
#     return cv2.bitwise_not(imagen)

# def remover_ruido(imagen):
#     return cv2.medianBlur(imagen, 3)

# # Cargar la imagen
# imagen_original = cv2.imread('C:/Users/amnv2/Desktop/prueba/energy_meter_10.jpg')

# # Aplicar umbralización
# imagen_umbralizada = aplicar_umbralizacion(imagen_original)

# # Aplicar filtrado gaussiano y morfológico
# imagen_preprocesada = aplicar_filtrado_morfo(imagen_umbralizada)

# # Recortar y aislar caracteres
# caracteres_recortados = recortar_caracteres(imagen_preprocesada)

# # Parámetros de escala deseada para normalización
# ancho_deseado, alto_deseado = 28, 28

# # Aplicar escalamiento y normalización a cada carácter
# caracteres_preprocesados = [escalar_normalizar(caracter, ancho_deseado, alto_deseado) for caracter in caracteres_recortados]

# # Invertir colores
# imagen_invertida = invertir_colores(imagen_preprocesada)

# # Remover ruido
# imagen_sin_ruido = remover_ruido(imagen_invertida)

# # Mostrar imágenes preprocesadas (opcional)
# cv2.imshow('Imagen Original', imagen_original)
# cv2.imshow('Imagen Umbralizada', imagen_umbralizada)
# cv2.imshow('Imagen Preprocesada', imagen_preprocesada)
# cv2.imshow('Imagen Invertida', imagen_invertida)
# cv2.imshow('Imagen Sin Ruido', imagen_sin_ruido)

# # Esperar hasta que se presione una tecla
# cv2.waitKey(0)
# cv2.destroyAllWindows()