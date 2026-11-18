import cv2
import numpy as np
import os

# -----------------------------------------
# 1. RUTAS Y CONFIGURACION
# -----------------------------------------

BASE = os.path.dirname(os.path.abspath(__file__))

CARPETA_ROSTROS = os.path.join(BASE, "Rostros", "Usuario")
CARPETA_MODELOS = os.path.join(BASE, "Modelos")

MODELO_YUNET = os.path.join(CARPETA_MODELOS, "yunet.onnx")
MODELO_SFACE = os.path.join(CARPETA_MODELOS, "sface.onnx")

CAMARA = 1

# Umbral inicial de reconocimiento
UMBRAL = 0.40

# -----------------------------------------
# 2. COMPROBAR ARCHIVOS
# -----------------------------------------

for archivo in [MODELO_YUNET, MODELO_SFACE]:

    if not os.path.isfile(archivo):
        raise FileNotFoundError(
            "No se encontro el modelo: " + archivo
        )

if not os.path.isdir(CARPETA_ROSTROS):
    raise FileNotFoundError("No existe la carpeta de rostros")

# -----------------------------------------
# 3. CARGAR MODELOS
# -----------------------------------------

detector = cv2.FaceDetectorYN.create(
    MODELO_YUNET,
    "",
    (320, 320),
    0.8,
    0.3,
    5000
)

reconocedor = cv2.FaceRecognizerSF.create(
    MODELO_SFACE,
    ""
)

print("Modelos cargados correctamente")

# -----------------------------------------
# 4. REGISTRAR TUS FOTOGRAFIAS
# -----------------------------------------

referencias = []

for archivo in os.listdir(CARPETA_ROSTROS):

    if not archivo.lower().endswith((".jpg", ".jpeg", ".png")):
        continue

    ruta = os.path.join(CARPETA_ROSTROS, archivo)

    imagen = cv2.imread(ruta)

    if imagen is None:
        print("No se pudo cargar:", archivo)
        continue

    alto, ancho = imagen.shape[:2]

    detector.setInputSize((ancho, alto))

    _, rostros = detector.detect(imagen)

    if rostros is None or len(rostros) == 0:
        print("No se encontro rostro en:", archivo)
        continue

    # Elegir el rostro de mayor tamaño
    rostro = max(
        rostros,
        key=lambda f: f[2] * f[3]
    )

    alineado = reconocedor.alignCrop(imagen, rostro)

    caracteristicas = reconocedor.feature(alineado)

    caracteristicas = caracteristicas.flatten()

    # Normalizar caracteristicas
    caracteristicas = caracteristicas / np.linalg.norm(caracteristicas)

    referencias.append(caracteristicas)

    print("Referencia registrada:", archivo)

if len(referencias) == 0:
    raise RuntimeError("No se pudieron registrar fotografias")

# Crear una referencia promedio
referencia = np.mean(referencias, axis=0)

referencia = referencia / np.linalg.norm(referencia)

print("Registro facial completado")
print("Fotografias utilizadas:", len(referencias))

# -----------------------------------------
# 5. INICIAR CAMARA
# -----------------------------------------

cap = cv2.VideoCapture(CAMARA)

if not cap.isOpened():
    raise RuntimeError("No se pudo abrir la camara")

print("Camara iniciada")

# -----------------------------------------
# 6. RECONOCIMIENTO EN TIEMPO REAL
# -----------------------------------------

while True:

    ret, frame = cap.read()

    if not ret:
        print("Error al leer la camara")
        break

    alto, ancho = frame.shape[:2]

    detector.setInputSize((ancho, alto))

    _, rostros = detector.detect(frame)

    if rostros is not None:

        for rostro in rostros:

            x, y, w, h = rostro[:4].astype(int)

            # Alinear rostro detectado
            alineado = reconocedor.alignCrop(
                frame,
                rostro
            )

            # Extraer caracteristicas
            caracteristicas = reconocedor.feature(
                alineado
            ).flatten()

            caracteristicas = caracteristicas / np.linalg.norm(
                caracteristicas
            )

            # Comparar con la referencia
            similitud = float(
                np.dot(referencia, caracteristicas)
            )

            # Identificar persona
            if similitud >= UMBRAL:

                nombre = "Jorge"
                color = (0, 255, 0)

            else:

                nombre = "Desconocido"
                color = (0, 0, 255)

            # Dibujar rectangulo
            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                color,
                2
            )

            texto = f"{nombre} ({similitud * 100:.1f}%)"

            cv2.putText(
                frame,
                texto,
                (x, max(25, y - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2
            )

    cv2.imshow("Reconocimiento Facial", frame)

    # ESC para salir
    if cv2.waitKey(1) & 0xFF == 27:
        break

# -----------------------------------------
# 7. CERRAR PROGRAMA
# -----------------------------------------

cap.release()
cv2.destroyAllWindows()

print("Programa finalizado")
input("Presiona ENTER para cerrar...")