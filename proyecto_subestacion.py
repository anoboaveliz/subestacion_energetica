import os
import re
import cv2
import torch
import datetime as dt
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

NUM_FOTOS = 5                                                                      # Definir la cantidad de fotos a tomar
SEGUNDOS_NUEVAS_FOTOS = 5                                                          # Definir cada cuanto tiempo la camara debe tomar fotos
MODEL_PATH = '/home/integradora/yolov5/weights/energy/best_energy.pt'              # Ruta al archivo .pt del modelo YOLOv5
DIRECTORIO_CAPTURAS_MEDIDOR = "/home/integradora/subestacion/fotos/originales/"     # DIRECTORIO donde se guardaran las imagenes originales
DIRECTORIO_IMAGENES_RECORTADAS = "/home/integradora/subestacion/fotos/recortadas/"  # DIRECTORIO donde se guardaran las imagenes recortadas
DIRECTORIO_IMAGENES_PROCESADAS = "/home/integradora/subestacion/fotos/procesadas/"  # DIRECTORIO donde se guardaran las imagenes procesadas
PROCESADOR_DE_TEXTO = "/home/integradora/subestacion/modelTRCOR/processor/"        # Ubicacion del modelo entrenado para reconocimiento de texto
DECODIFICADOR = "/home/integradora/subestacion/modelTRCOR/decoder/"                # Ubicacion del archivo decodificador de texto
CODIFICADOR = "/home/integradora/subestacion/modelTRCOR/encoder/"                  # Ubicacion del archivo codificador de texto

def ordenar_archivos_por_numero(lista_archivos):
    # Función para extraer el número del nombre del archivo
    def obtener_numero(nombre_archivo):
        match = re.search(r'\d+', nombre_archivo)
        return int(match.group()) if match else -1
    # Ordenar la lista de archivos utilizando la función obtener_numero como clave
    lista_ordenada = sorted(lista_archivos, key=obtener_numero)
    return lista_ordenada

def inicializar_sistema():
    # Asegurarse de que cada DIRECTORIO exista
    if not os.path.exists(DIRECTORIO_CAPTURAS_MEDIDOR):
        os.makedirs(DIRECTORIO_CAPTURAS_MEDIDOR)
    if not os.path.exists(DIRECTORIO_IMAGENES_RECORTADAS):
        os.makedirs(DIRECTORIO_IMAGENES_RECORTADAS)
    if not os.path.exists(DIRECTORIO_IMAGENES_PROCESADAS):
        os.makedirs(DIRECTORIO_IMAGENES_PROCESADAS)
    # Obtener la lista de archivos en el directorio
    archivos_existentes = os.listdir(DIRECTORIO_CAPTURAS_MEDIDOR)
    # Obtener el valor inicial del contador
    contador = len(archivos_existentes) + 1
    # Cargar modelo torch, procesador de texto en imagenes y modelo entrenado
    model_torch = torch.hub.load('ultralytics/yolov5', 'custom', path = MODEL_PATH)
    processor = TrOCRProcessor.from_pretrained(PROCESADOR_DE_TEXTO)
    model_cod_decod = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(CODIFICADOR, DECODIFICADOR)
    # Establecer tiempo actual
    tiempo_inicial = dt.datetime.now()
    return tiempo_inicial, contador, processor, model_torch, model_cod_decod

def tomar_fotos(tiempoA, numero_fotos_a_tomar ,numero_fotos_existentes):
    # Crear instancia de la camara USB
    cap = cv2.VideoCapture(0)
    # Verificar si la camara se ha activado correctamente
    if not cap.isOpened():
        print("Error al abrir la camara")
        exit()
    # Tomar las fotos
    while numero_fotos_a_tomar > 0:
        # Leer un fotograma del flujo de video
        ret, frame = cap.read()

        # Verificar si se ha leido correctamente el fotograma
        if not ret:
            print("Error al leer el fotograma")
            break

        if ret:
            # Almacenar el tiempo actual
            tiempoB = dt.datetime.now()
            # Cuanto tiempo ha pasado desde tiempoA?
            tiempoTranscurrido = tiempoB - tiempoA
            
            if tiempoTranscurrido.seconds >= SEGUNDOS_NUEVAS_FOTOS:
                # Generar el nombre de la imagen
                nombre_imagen = f"energy_meter_{numero_fotos_existentes}.jpg"
                ruta_imagen = os.path.join(DIRECTORIO_CAPTURAS_MEDIDOR, nombre_imagen)

                # Verificar si el archivo ya existe y buscar un nombre unico
                while os.path.exists(ruta_imagen):
                    numero_fotos_existentes += 1
                    nombre_imagen = f"energy_meter_{numero_fotos_existentes}.jpg"
                    ruta_imagen = os.path.join(DIRECTORIO_CAPTURAS_MEDIDOR, nombre_imagen)

                # Guardar la imagen en un archivo
                cv2.imwrite(ruta_imagen, frame)
                print(f"Imagen {numero_fotos_existentes} guardada correctamente")
                # Incrementar el contador
                numero_fotos_existentes += 1
                tiempoA = dt.datetime.now()
                numero_fotos_a_tomar -= 1

            if cv2.waitKey(1) == ord('s'):
                break
        else:
            break
    # Liberar la camara
    cap.release()

def procesamiento_imagenes (modelo_torch, dir_recortadas, dir_procesadas):
    # Cargar el modelo YOLOv5
    count=0 
    process=0
    if os.path.exists(DIRECTORIO_CAPTURAS_MEDIDOR):
        for nombre_archivo in os.listdir(DIRECTORIO_CAPTURAS_MEDIDOR):
            ruta_archivo = os.path.join(DIRECTORIO_CAPTURAS_MEDIDOR, nombre_archivo)
            if os.path.isfile(ruta_archivo):
                try:
                    # Cargar la imagen de entrada
                    image = cv2.imread(ruta_archivo)
                    # Realizar la deteccion de objetos con YOLOv5
                    results = modelo_torch(image)
                    # Obtener las coordenadas y las etiquetas de los objetos detectados
                    pred = results.pandas().xyxy[0]  # Obtener las predicciones en formato pandas
                    boxes = pred[['xmin', 'ymin', 'xmax', 'ymax']].values
                    labels = pred['name'].tolist()

                    # Iterar sobre las regiones detectadas y extraer el texto utilizando EasyOCR
                    for box, label in zip(boxes, labels):
                        xmin, ymin, xmax, ymax = box
                        cropped_image = image[int(ymin):int(ymax), int(xmin):int(xmax)]
                        ruta_imagen = dir_recortadas + f'energy_meter_{count}.jpg'
                        cv2.imwrite(ruta_imagen, cropped_image)
                        count +=1 

                    # Dibujar los cuadros delimitadores y mostrar el texto detectado
                    for box in boxes:
                        xmin, ymin, xmax, ymax = box
                        cv2.rectangle(image, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 255, 0), 2)

                    # Guardar la imagen procesada
                    output_path = dir_procesadas + f'imagen_procesada_{process}.jpg'
                    cv2.imwrite(output_path, image)
                    process+=1
                except Exception as e:
                    print(f"Error al abrir {ruta_archivo}: {e}")
    else:
        print("El directorio no existe.")

def extraccion_de_texto(processor, model):
    if os.path.exists(DIRECTORIO_IMAGENES_RECORTADAS):
        for nombre_archivo in ordenar_archivos_por_numero(os.listdir(DIRECTORIO_IMAGENES_RECORTADAS)):
            ruta_archivo = os.path.join(DIRECTORIO_IMAGENES_RECORTADAS, nombre_archivo)
            if os.path.isfile(ruta_archivo):
                try:
                    print(nombre_archivo)
                    image = Image.open(ruta_archivo).convert("RGB")
                    pixel_values = processor(image, return_tensors="pt").pixel_values
                    #print(pixel_values.shape)
                    generated_ids = model.generate(pixel_values, max_new_tokens=1000)
                    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                    print(generated_text)
                except Exception as e:
                    print(f"Error al abrir {ruta_archivo}: {e}")
    else:
        print("El directorio no existe.")


def main():
    tiempo_inicial, fotos_existentes, processor, model_torch, model_cod_decod = inicializar_sistema()
    tomar_fotos(tiempo_inicial, NUM_FOTOS, fotos_existentes)
    procesamiento_imagenes(model_torch, DIRECTORIO_IMAGENES_RECORTADAS, DIRECTORIO_IMAGENES_PROCESADAS)
    extraccion_de_texto(processor, model_cod_decod)

if __name__ == "__main__":
    main()