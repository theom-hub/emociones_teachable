import cv2
import numpy as np
import tf_keras
from PIL import Image, ImageOps
import os


# -------------------------------------------------
# CONFIGURACIÓN
# -------------------------------------------------

MODELO = "modelo/keras_model.h5"
ETIQUETAS = "modelo/labels.txt"

UMBRAL = 0.80


# -------------------------------------------------
# CARGAR MODELO
# -------------------------------------------------

print("Cargando modelo...")

modelo = tf_keras.models.load_model(
    MODELO,
    compile=False
)

print("Modelo cargado.")


# -------------------------------------------------
# CARGAR ETIQUETAS
# -------------------------------------------------

etiquetas = []

with open(ETIQUETAS, "r", encoding="utf-8") as archivo:

    for linea in archivo:

        linea = linea.strip()

        # Teachable Machine guarda:
        # 0 felicidad
        # 1 tristeza
        # etc.

        partes = linea.split(" ", 1)

        etiquetas.append(partes[-1])


print("Clases encontradas:")

for i, etiqueta in enumerate(etiquetas):
    print(i, "->", etiqueta)


# -------------------------------------------------
# CARGAR IMÁGENES
# -------------------------------------------------

imagenes_emociones = {}

for emocion in etiquetas:

    ruta = os.path.join(
        "imagenes",
        f"{emocion}.jpg"
    )

    imagen = cv2.imread(ruta)

    if imagen is None:
        print("No encontré:", ruta)

    else:
        imagenes_emociones[emocion] = imagen


# -------------------------------------------------
# PREPARAR IMAGEN
# -------------------------------------------------

def preparar_imagen(frame):

    # OpenCV trabaja en BGR.
    # Nuestro modelo necesita RGB.

    imagen_rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    imagen = Image.fromarray(imagen_rgb)

    imagen = ImageOps.fit(
        imagen,
        (224, 224),
        method=Image.Resampling.LANCZOS
    )

    array = np.asarray(
        imagen
    ).astype(np.float32)

    array = (array / 127.5) - 1

    datos = np.expand_dims(
        array,
        axis=0
    )

    return datos


# -------------------------------------------------
# ABRIR CÁMARA
# -------------------------------------------------

camara = cv2.VideoCapture(0)

if not camara.isOpened():
    print("No se pudo abrir la cámara.")
    exit()

print("Reconocimiento iniciado.")
print("Presioná Q para salir.")


# -------------------------------------------------
# IMAGEN INICIAL
# -------------------------------------------------

imagen_actual = np.zeros(
    (480, 480, 3),
    dtype=np.uint8
)

cv2.putText(
    imagen_actual,
    "Hace una expresion",
    (90, 240),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.8,
    (255, 255, 255),
    2
)


# -------------------------------------------------
# BUCLE PRINCIPAL
# -------------------------------------------------

while True:

    correcto, frame = camara.read()

    if not correcto:
        break


    # Preparar imagen

    datos = preparar_imagen(frame)


    # Hacer predicción

    predicciones = modelo.predict(
        datos,
        verbose=0
    )[0]

    indice = np.argmax(predicciones)

    confianza = predicciones[indice]

    emocion = etiquetas[indice]


    # Comprobar confianza

    if confianza >= UMBRAL:

        texto = f"{emocion} {confianza * 100:.0f}%"

        color = (0, 255, 0)

        if emocion in imagenes_emociones:

            imagen_actual = (
                imagenes_emociones[emocion].copy()
            )

    else:

        texto = (
            f"No estoy seguro "
            f"{confianza * 100:.0f}%"
        )

        color = (0, 0, 255)


    # Mostrar resultado

    cv2.putText(
        frame,
        texto,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        color,
        2
    )


    # Unir cámara e imagen de la emoción

    frame_mostrar = cv2.resize(
        frame,
        (480, 480)
    )

    imagen_mostrar = cv2.resize(
        imagen_actual,
        (480, 480)
    )

    pantalla = np.hstack(
        (frame_mostrar, imagen_mostrar)
    )


    cv2.imshow(
        "Detector de emociones",
        pantalla
    )


    # Salir con Q

    tecla = cv2.waitKey(30) & 0xFF

    if tecla == ord("q") or tecla == ord("Q"):
        break


# -------------------------------------------------
# CERRAR PROGRAMA
# -------------------------------------------------

camara.release()
cv2.destroyAllWindows()

print("Programa terminado.")