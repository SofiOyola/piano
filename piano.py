import os
import sys
import warnings

# Evitar conflicto entre TensorFlow y Protobuf al importar MediaPipe Hands
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
sys.modules["tensorflow"] = None

# Suprimir advertencia protobuf temprano
warnings.filterwarnings("ignore", message=".*SymbolDatabase.GetPrototype.*")

import cv2
import mediapipe as mp
import numpy as np
import pygame
import time
import tempfile
import atexit
import shutil
from collections import deque

# Intentaremos usar pydub solo si está disponible para convertir mp3->wav
try:
    from pydub import AudioSegment
    PYDUB_AVAILABLE = True
except Exception:
    PYDUB_AVAILABLE = False

# --------------------------
# CONFIGURACIÓN DE NOTAS (Formato Piano: Do, Re, Mi, Fa, Sol, La, Si)
# --------------------------
# Archivos de audio (en carpeta sonidos/) ordenados de izquierda a derecha como en un piano:
# Mano 1 (5 dedos de izq a der): Do, Re, Mi, Fa, Sol (C4..G4)
# Mano 2 (5 dedos de izq a der): La, Si, Do, Re, Mi  (A4..E5)
archivos_por_dedo = ['C4', 'D4', 'E4', 'F4', 'G4', 'A4', 'B4', 'C5', 'D5', 'E5']

# Nombres en formato piano tradicional para mostrar en los dedos y en el pentagrama
nombres_piano_por_dedo = ['Do', 'Re', 'Mi', 'Fa', 'Sol', 'La', 'Si', 'Do', 'Re', 'Mi']

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ruta_sonidos = os.path.join(BASE_DIR, "sonidos")

FRECUENCIAS_NOTAS = {
    'C4': 261.63, 'D4': 293.66, 'E4': 329.63, 'F4': 349.23, 'G4': 392.00,
    'A4': 440.00, 'B4': 493.88, 'C5': 523.25, 'D5': 587.33, 'E5': 659.25
}

# --------------------------
# PYGAME / CARGA SAMPLES
# --------------------------
try:
    pygame.mixer.pre_init(44100, -16, 2, 512)
    pygame.mixer.init()
    pygame.mixer.set_num_channels(16)
except Exception as e:
    print("⚠️ pygame.mixer.init() falló:", e)
    raise

TMP_DIR = tempfile.mkdtemp(prefix="piano_sounds_")
atexit.register(lambda: shutil.rmtree(TMP_DIR, ignore_errors=True))

def ffmpeg_available():
    return shutil.which("ffmpeg") is not None or shutil.which("ffmpeg.exe") is not None

def try_load(path):
    try:
        return pygame.mixer.Sound(path)
    except Exception:
        return None

def sintetizar_tono_piano(freq, duration=0.6, sample_rate=44100):
    """Genera un tono tipo piano acústico como respaldo si algún archivo no carga."""
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    envelope = np.exp(-3.5 * t) * (1 - np.exp(-80 * t))
    wave = (
        1.0 * np.sin(2 * np.pi * freq * t) +
        0.4 * np.sin(2 * np.pi * (2 * freq) * t) +
        0.15 * np.sin(2 * np.pi * (3 * freq) * t) +
        0.05 * np.sin(2 * np.pi * (4 * freq) * t)
    ) * envelope
    wave = wave / np.max(np.abs(wave)) * 0.6
    audio_int16 = (wave * 32767).astype(np.int16)
    stereo = np.column_stack((audio_int16, audio_int16))
    return pygame.sndarray.make_sound(stereo)

sonidos = {}
for i, codigo_archivo in enumerate(archivos_por_dedo):
    mp3_path = os.path.join(ruta_sonidos, f"{codigo_archivo}.mp3")
    wav_path = os.path.join(ruta_sonidos, f"{codigo_archivo}.wav")
    loaded = False

    if os.path.exists(mp3_path):
        snd = try_load(mp3_path)
        if snd:
            sonidos[i] = snd
            loaded = True
        else:
            if PYDUB_AVAILABLE and ffmpeg_available():
                try:
                    audio = AudioSegment.from_file(mp3_path, format="mp3")
                    out_wav = os.path.join(TMP_DIR, f"{codigo_archivo}.wav")
                    audio.export(out_wav, format="wav")
                    snd = try_load(out_wav)
                    if snd:
                        sonidos[i] = snd
                        loaded = True
                except Exception as e:
                    print(f"⚠️ Error convirtiendo {mp3_path} a WAV: {e}")

    if not loaded and os.path.exists(wav_path):
        snd = try_load(wav_path)
        if snd:
            sonidos[i] = snd
            loaded = True

    if not loaded:
        sonidos[i] = sintetizar_tono_piano(FRECUENCIAS_NOTAS.get(codigo_archivo, 440.0))

# --------------------------
# MEDIAPIPE HANDS
# --------------------------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    model_complexity=1,
    min_detection_confidence=0.55,
    min_tracking_confidence=0.50
)

# Índices de las 5 puntas y sus articulaciones inmediatas
FINGER_TIPS = [4, 8, 12, 16, 20]
FINGER_PIPS = [3, 6, 10, 14, 18]

# --------------------------
# DETECCIÓN DE PULSACIÓN PARA MANO HORIZONTAL / ARAÑA
# --------------------------
# En postura horizontal (araña), los 5 dedos de cada mano se ordenan espacialmente
# de izquierda a derecha en X para que las notas nunca se crucen ni se inviertan.
# Cuando un dedo baja (dy > UMBRAL), restamos el movimiento mínimo del resto de la mano:
#   - Si toda la mano baja al apoyarse, todos los dedos bajan igual -> NO suena por error.
#   - Si bajas un dedo para tocar una nota, ese dedo baja más que el resto -> SUENA inmediato.

alpha_smooth = 0.65            # Suavizado EMA de posición
UMBRAL_MOV_REL = 0.014         # Umbral de bajada por dedo (sensible para mano horizontal)
COOLDOWN = 0.20                # Segundos entre repeticiones del mismo dedo
DURACION_NOTA_VISUAL = 0.65    # Tiempo (s) que el círculo permanece iluminado en el pentagrama

prev_y = [None] * 10
prev_x = [None] * 10
hist_y = [deque(maxlen=3) for _ in range(10)]
dedo_abajo = [False] * 10
last_play = {i: 0.0 for i in range(10)}
ultimo_trigger_visual = {i: 0.0 for i in range(10)}

def reset_finger(idx):
    prev_y[idx] = None
    prev_x[idx] = None
    hist_y[idx].clear()
    dedo_abajo[idx] = False

def tocar_sonido_idx(idx):
    if idx not in sonidos:
        return False
    ahora = time.time()
    if ahora - last_play.get(idx, 0.0) < COOLDOWN:
        return False
    try:
        sonidos[idx].play()
        last_play[idx] = ahora
        ultimo_trigger_visual[idx] = ahora
        return True
    except Exception as e:
        print("Error reproduciendo sonido:", e)
        return False

# --------------------------
# OBTENER LAS PUNTAS ORDENADAS DE IZQUIERDA A DERECHA EN HORIZONTAL
# --------------------------
def extraer_5_puntas_ordenadas_x(lm, frame_width, frame_height):
    """
    Extrae las 5 puntas de una mano y las ordena de IZQUIERDA a DERECHA en pantalla (eje X).
    Esto evita que MediaPipe cruce los dedos o ponga 'Do' debajo de 'Mi' cuando la mano
    se coloca horizontalmente o en forma de araña con la palma hacia abajo.
    """
    puntas = []
    for k in range(5):
        tip_id = FINGER_TIPS[k]
        pip_id = FINGER_PIPS[k]
        x_norm = lm.landmark[tip_id].x
        y_norm = lm.landmark[tip_id].y
        x_px = int(x_norm * frame_width)
        y_px = int(y_norm * frame_height)
        pip_px = (int(lm.landmark[pip_id].x * frame_width), int(lm.landmark[pip_id].y * frame_height))
        puntas.append({
            "x_norm": x_norm,
            "y_norm": y_norm,
            "x_px": x_px,
            "y_px": y_px,
            "pip_px": pip_px
        })

    # Ordenar estrictamente de izquierda a derecha en X
    puntas.sort(key=lambda p: p["x_norm"])
    return puntas

def obtener_tips_ordenados(results, frame_width, frame_height):
    """
    Retorna lista de 10 entradas (una por dedo).
    - Si hay 1 mano en cámara: usa los primeros 5 dedos (Do, Re, Mi, Fa, Sol) de izq a der.
    - Si hay 2 manos en cámara:
        Mano izquierda en pantalla -> Do, Re, Mi, Fa, Sol (índices 0..4)
        Mano derecha en pantalla   -> La, Si, Do, Re, Mi  (índices 5..9)
    """
    if not results.multi_hand_landmarks:
        return [None] * 10

    manos = []
    for lm in results.multi_hand_landmarks:
        cx = float(np.mean([lm.landmark[t].x for t in FINGER_TIPS]))
        manos.append((cx, lm))

    # Ordenar las manos detectadas de izquierda a derecha en pantalla
    manos.sort(key=lambda item: item[0])

    salida = [None] * 10
    if len(manos) == 1:
        # Con 1 mano en escena, mapeamos a Do, Re, Mi, Fa, Sol (0..4)
        puntas_m1 = extraer_5_puntas_ordenadas_x(manos[0][1], frame_width, frame_height)
        for j in range(5):
            salida[j] = puntas_m1[j]
    else:
        # Mano más a la izquierda -> 0..4 (Do, Re, Mi, Fa, Sol)
        puntas_m1 = extraer_5_puntas_ordenadas_x(manos[0][1], frame_width, frame_height)
        # Mano más a la derecha -> 5..9 (La, Si, Do, Re, Mi)
        puntas_m2 = extraer_5_puntas_ordenadas_x(manos[1][1], frame_width, frame_height)
        for j in range(5):
            salida[j] = puntas_m1[j]
            salida[j + 5] = puntas_m2[j]

    return salida

# --------------------------
# PENTAGRAMA EN LA ESQUINA SUPERIOR DERECHA
# --------------------------
def dibujar_pentagrama(frame, ahora):
    """
    Dibuja un pentagrama musical en la parte superior derecha de la pantalla
    y muestra un círculo sobre la línea o espacio exacto de la nota que se está tocando.

    Posiciones en Clave de Sol (en medios pasos desde la 5ª línea superior):
      Línea 5 (arriba) : paso 0  (Fa5)
      Espacio 4        : paso 1  -> Mi (E5, dedo 9)
      Línea 4          : paso 2  -> Re (D5, dedo 8)
      Espacio 3        : paso 3  -> Do (C5, dedo 7)
      Línea 3 (central): paso 4  -> Si (B4, dedo 6)
      Espacio 2        : paso 5  -> La (A4, dedo 5)
      Línea 2          : paso 6  -> Sol (G4, dedo 4)
      Espacio 1        : paso 7  -> Fa (F4, dedo 3)
      Línea 1 (abajo)  : paso 8  -> Mi (E4, dedo 2)
      Espacio inferior : paso 9  -> Re (D4, dedo 1)
      Línea adic. inf. : paso 10 -> Do (C4, dedo 0)
    """
    h, w, _ = frame.shape

    # Dimensiones del recuadro del pentagrama (arriba a la derecha)
    box_w = min(330, w - 20)
    box_h = 145
    x1 = w - box_w - 12
    y1 = 12
    x2 = w - 12
    y2 = y1 + box_h

    # Fondo semitransparente tipo papel de partitura
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (248, 246, 238), -1)
    cv2.addWeighted(overlay, 0.88, frame, 0.12, 0, frame)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (60, 60, 60), 2)

    # Título del pentagrama y notas activas
    notas_activas_nombres = [
        nombres_piano_por_dedo[i]
        for i in range(10)
        if (ahora - ultimo_trigger_visual.get(i, 0.0)) < DURACION_NOTA_VISUAL
    ]
    texto_titulo = "Pentagrama"
    if notas_activas_nombres:
        texto_titulo += f" - Tocando: {', '.join(notas_activas_nombres)}"
    cv2.putText(
        frame, texto_titulo, (x1 + 10, y1 + 20),
        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (30, 30, 30), 1, cv2.LINE_AA
    )

    # Geometría de las 5 líneas del pentagrama
    staff_left = x1 + 42
    staff_right = x2 - 14
    top_line_y = y1 + 38
    line_spacing = 14          # Distancia entre líneas del pentagrama
    half_step = line_spacing // 2  # 7 px entre línea y espacio

    # Dibujar las 5 líneas horizontales del pentagrama (Línea 5 arriba -> Línea 1 abajo)
    for l_idx in range(5):
        ly = top_line_y + l_idx * line_spacing
        cv2.line(frame, (staff_left - 26, ly), (staff_right, ly), (50, 50, 50), 1, cv2.LINE_AA)

    # Barras de compás al inicio y final del pentagrama
    bottom_line_y = top_line_y + 4 * line_spacing
    cv2.line(frame, (staff_left - 26, top_line_y), (staff_left - 26, bottom_line_y), (40, 40, 40), 2)
    cv2.line(frame, (staff_right, top_line_y), (staff_right, bottom_line_y), (40, 40, 40), 2)

    # Símbolo estilizado de Clave de Sol al inicio
    clef_x = staff_left - 14
    sol_line_y = top_line_y + 3 * line_spacing
    cv2.circle(frame, (clef_x, sol_line_y), 6, (40, 40, 40), 1, cv2.LINE_AA)
    cv2.line(frame, (clef_x + 3, top_line_y - 6), (clef_x - 2, bottom_line_y + 10), (40, 40, 40), 2, cv2.LINE_AA)
    cv2.circle(frame, (clef_x - 4, bottom_line_y + 10), 3, (40, 40, 40), -1, cv2.LINE_AA)

    # Pasos verticales desde la 5ª línea (0) hasta Do4 (10)
    pasos_por_dedo = [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
    ancho_util = staff_right - staff_left - 16

    for i in range(10):
        paso = pasos_por_dedo[i]
        nx = int(staff_left + 8 + (i * ancho_util / 9.0))
        ny = int(top_line_y + paso * half_step)
        nombre_nota = nombres_piano_por_dedo[i]
        activa = (ahora - ultimo_trigger_visual.get(i, 0.0)) < DURACION_NOTA_VISUAL

        # Guía tenue de las posiciones en el pentagrama
        if not activa:
            cv2.circle(frame, (nx, ny), 3, (175, 175, 175), -1, cv2.LINE_AA)
            cv2.putText(
                frame, nombre_nota, (nx - 8, y2 - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.34, (130, 130, 130), 1, cv2.LINE_AA
            )
        else:
            # Si es Do central (i == 0, paso 10), dibujar la línea adicional inferior
            if i == 0:
                cv2.line(frame, (nx - 12, ny), (nx + 12, ny), (30, 30, 30), 2, cv2.LINE_AA)

            # Círculo destacado en la nota que se está tocando en ese momento
            cv2.circle(frame, (nx, ny), 8, (0, 210, 70), -1, cv2.LINE_AA)
            cv2.circle(frame, (nx, ny), 8, (15, 70, 20), 2, cv2.LINE_AA)

            # Plica vertical de la figura musical
            if paso >= 4:
                cv2.line(frame, (nx + 7, ny), (nx + 7, ny - 24), (15, 70, 20), 2, cv2.LINE_AA)
            else:
                cv2.line(frame, (nx - 7, ny), (nx - 7, ny + 24), (15, 70, 20), 2, cv2.LINE_AA)

            # Nombre de la nota resaltado
            cv2.putText(
                frame, nombre_nota, (nx - 10, y2 - 7),
                cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0, 150, 40), 2, cv2.LINE_AA
            )

# --------------------------
# MENÚ DE INICIO (BOTÓN PIANO "START") Y BOTÓN "SALIR"
# --------------------------
estado_ui = {
    "en_menu": True,
    "salir": False,
    "mouse_x": 0,
    "mouse_y": 0,
    "start_rect": (0, 0, 0, 0),  # (x1, y1, x2, y2)
    "salir_rect": (0, 0, 0, 0),  # (x1, y1, x2, y2)
}

def punto_en_rect(px, py, rect):
    x1, y1, x2, y2 = rect
    return x1 <= px <= x2 and y1 <= py <= y2

def mouse_callback(event, x, y, flags, param):
    estado_ui["mouse_x"] = x
    estado_ui["mouse_y"] = y
    if event == cv2.EVENT_LBUTTONDOWN:
        # Clic en botón Salir (abajo a la derecha)
        if punto_en_rect(x, y, estado_ui["salir_rect"]):
            estado_ui["salir"] = True
            return
        # Clic en botón START con forma de piano (cuando estamos en el menú)
        if estado_ui["en_menu"] and punto_en_rect(x, y, estado_ui["start_rect"]):
            estado_ui["en_menu"] = False
            for i in range(10):
                reset_finger(i)
            tocar_sonido_idx(0)  # Sonido Do de bienvenida

def dibujar_boton_salir(frame):
    """
    Dibuja el botón 'Salir' en la esquina inferior derecha de la pantalla.
    """
    h, w, _ = frame.shape
    btn_w, btn_h = 115, 40
    x1 = w - btn_w - 14
    y1 = h - btn_h - 14
    x2 = w - 14
    y2 = h - 14
    estado_ui["salir_rect"] = (x1, y1, x2, y2)

    hover = punto_en_rect(estado_ui["mouse_x"], estado_ui["mouse_y"], estado_ui["salir_rect"])
    color_fondo = (45, 45, 220) if hover else (35, 35, 175)

    cv2.rectangle(frame, (x1, y1), (x2, y2), color_fondo, -1, cv2.LINE_AA)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (240, 240, 240), 2, cv2.LINE_AA)
    cv2.putText(
        frame, "Salir",
        (x1 + 30, y1 + 27),
        cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2, cv2.LINE_AA
    )

def dibujar_menu_inicio(frame):
    """
    Dibuja el menú inicial con un botón central de 'START' que tiene forma de piano.
    """
    h, w, _ = frame.shape

    # Oscurecer suavemente el fondo del video para destacar el menú
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, h), (18, 18, 24), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

    # Título superior
    titulo = "PIANO VIRTUAL"
    (tw, _), _ = cv2.getTextSize(titulo, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)
    cv2.putText(
        frame, titulo,
        ((w - tw) // 2, int(h * 0.22)),
        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3, cv2.LINE_AA
    )

    subtitulo = "Haz clic en el piano START para comenzar"
    (sw, _), _ = cv2.getTextSize(subtitulo, cv2.FONT_HERSHEY_SIMPLEX, 0.60, 1)
    cv2.putText(
        frame, subtitulo,
        ((w - sw) // 2, int(h * 0.29)),
        cv2.FONT_HERSHEY_SIMPLEX, 0.60, (200, 230, 255), 1, cv2.LINE_AA
    )

    # Dimensiones del botón START con forma de piano (7 teclas blancas + 5 negras)
    num_teclas_blancas = 7
    ancho_tecla = 40
    piano_w = num_teclas_blancas * ancho_tecla  # 280 px
    piano_h = 145
    px1 = (w - piano_w) // 2
    py1 = (h - piano_h) // 2 + 15
    px2 = px1 + piano_w
    py2 = py1 + piano_h
    estado_ui["start_rect"] = (px1 - 8, py1 - 18, px2 + 8, py2 + 8)

    hover = punto_en_rect(estado_ui["mouse_x"], estado_ui["mouse_y"], estado_ui["start_rect"])

    # Caja exterior del piano (madera/negro de piano con borde brillante al pasar el mouse)
    color_borde_piano = (0, 230, 100) if hover else (80, 190, 255)
    cv2.rectangle(frame, (px1 - 8, py1 - 18), (px2 + 8, py2 + 8), (25, 25, 25), -1, cv2.LINE_AA)
    cv2.rectangle(frame, (px1 - 8, py1 - 18), (px2 + 8, py2 + 8), color_borde_piano, 3, cv2.LINE_AA)
    # Filete clásico rojo/dorado sobre las teclas
    cv2.line(frame, (px1, py1 - 3), (px2, py1 - 3), (40, 40, 180), 3, cv2.LINE_AA)

    # 1. Dibujar las 7 teclas blancas
    for k in range(num_teclas_blancas):
        kx1 = px1 + k * ancho_tecla
        kx2 = kx1 + ancho_tecla
        color_blanca = (255, 255, 255) if not hover else (240, 255, 245)
        cv2.rectangle(frame, (kx1 + 1, py1), (kx2 - 1, py2), color_blanca, -1, cv2.LINE_AA)
        # Sombra inferior 3D de cada tecla blanca
        cv2.rectangle(frame, (kx1 + 1, py2 - 10), (kx2 - 1, py2), (210, 210, 210), -1, cv2.LINE_AA)
        cv2.rectangle(frame, (kx1, py1), (kx2, py2), (40, 40, 40), 1, cv2.LINE_AA)

    # 2. Dibujar las 5 teclas negras (entre Do-Re, Re-Mi | Fa-Sol, Sol-La, La-Si)
    ancho_negra = 24
    alto_negra = int(piano_h * 0.56)
    posiciones_negras = [0, 1, 3, 4, 5]
    for k in posiciones_negras:
        nx1 = px1 + (k + 1) * ancho_tecla - (ancho_negra // 2)
        nx2 = nx1 + ancho_negra
        ny2 = py1 + alto_negra
        cv2.rectangle(frame, (nx1, py1), (nx2, ny2), (20, 20, 20), -1, cv2.LINE_AA)
        cv2.rectangle(frame, (nx1 + 3, py1 + 4), (nx2 - 3, ny2 - 6), (55, 55, 55), 1, cv2.LINE_AA)

    # 3. Placa central/inferior con el texto "START" sobre las teclas del piano
    badge_w = 170
    badge_h = 42
    bx1 = (w - badge_w) // 2
    by1 = py2 - badge_h - 14
    bx2 = bx1 + badge_w
    by2 = by1 + badge_h

    color_badge = (0, 185, 70) if hover else (20, 30, 45)
    cv2.rectangle(frame, (bx1, by1), (bx2, by2), color_badge, -1, cv2.LINE_AA)
    cv2.rectangle(frame, (bx1, by1), (bx2, by2), (255, 255, 255), 2, cv2.LINE_AA)

    txt_start = "START"
    (stw, sth), _ = cv2.getTextSize(txt_start, cv2.FONT_HERSHEY_SIMPLEX, 0.95, 2)
    cv2.putText(
        frame, txt_start,
        (bx1 + (badge_w - stw) // 2, by1 + (badge_h + sth) // 2 - 2),
        cv2.FONT_HERSHEY_SIMPLEX, 0.95, (255, 255, 255), 2, cv2.LINE_AA
    )

# --------------------------
# BUCLE PRINCIPAL
# --------------------------
def main():
    global UMBRAL_MOV_REL

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("No se puede abrir la cámara.")
        return

    nombre_ventana = "Piano Virtual - Mano Horizontal + Pentagrama"
    cv2.namedWindow(nombre_ventana)
    cv2.setMouseCallback(nombre_ventana, mouse_callback)

    print("\n🎹 Piano con las Manos Activado (Do, Re, Mi, Fa, Sol, La, Si)")
    print("• Haz clic en el botón de piano 'START' para iniciar.")
    print("• Haz clic en el botón 'Salir' (abajo a la derecha) o presiona [ESC] para cerrar.\n")

    while not estado_ui["salir"]:
        ret, frame = cap.read()
        if not ret:
            print("Error leyendo frame.")
            break

        # Vista espejo natural
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        ahora = time.time()

        if estado_ui["en_menu"]:
            # Mostrar menú inicial con el botón START en forma de piano y el botón Salir
            dibujar_menu_inicio(frame)
            dibujar_boton_salir(frame)
        else:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            if results.multi_hand_landmarks:
                tips = obtener_tips_ordenados(results, w, h)

                # Procesar por cada mano (0..4 y 5..9)
                for mano_offset in (0, 5):
                    indices_mano = range(mano_offset, mano_offset + 5)
                    dys_mano = {}

                    for i in indices_mano:
                        info = tips[i]
                        if info is None:
                            reset_finger(i)
                            continue

                        y_norm = info["y_norm"]
                        x_px = info["x_px"]
                        y_px = info["y_px"]

                        # Suavizado EMA en X e Y
                        if prev_y[i] is None:
                            y_smooth = y_norm
                            x_smooth = float(x_px)
                        else:
                            y_smooth = alpha_smooth * y_norm + (1.0 - alpha_smooth) * prev_y[i]
                            x_smooth = alpha_smooth * float(x_px) + (1.0 - alpha_smooth) * prev_x[i]

                        hist_y[i].append(y_smooth)

                        # Calcular bajada en 1 y 2 frames para detectar el toque al bajar un dedo
                        dy_1 = (y_smooth - prev_y[i]) if prev_y[i] is not None else 0.0
                        dy_2 = (y_smooth - hist_y[i][0]) * 0.65 if len(hist_y[i]) >= 2 else dy_1
                        dy_dedo = max(dy_1, dy_2)

                        dys_mano[i] = (dy_dedo, int(x_smooth), int(y_px), info["pip_px"])
                        prev_y[i] = y_smooth
                        prev_x[i] = x_smooth

                    if not dys_mano:
                        continue

                    # Si toda la mano baja en bloque (los 5 dedos bajan juntos), restamos el mínimo
                    # para que acomodar la mano no dispare todas las notas a la vez.
                    # Pero si bajas 1 dedo (o varios) mientras al menos uno se mantiene, suena de inmediato.
                    valores_dy = [v[0] for v in dys_mano.values()]
                    bajada_comun_mano = max(0.0, min(valores_dy)) if len(valores_dy) >= 4 else 0.0

                    for i, (dy_dedo, draw_x, draw_y, pip_px) in dys_mano.items():
                        dy_efectivo = dy_dedo - bajada_comun_mano
                        nombre_nota = nombres_piano_por_dedo[i]

                        # Detectar bajada del dedo
                        if not dedo_abajo[i]:
                            if dy_efectivo > UMBRAL_MOV_REL:
                                tocar_sonido_idx(i)
                                dedo_abajo[i] = True
                        else:
                            # Liberar estado apenas el dedo deja de bajar
                            if dy_dedo < (UMBRAL_MOV_REL * 0.35):
                                dedo_abajo[i] = False

                        # Visualización limpia en cada punta de dedo (sin cruzar líneas en la palma)
                        reciente = (ahora - ultimo_trigger_visual.get(i, 0.0)) < 0.28
                        color_circulo = (0, 230, 60) if reciente else (255, 90, 0)

                        # Línea corta solo hacia la falange de ese mismo dedo para ver la articulación
                        cv2.line(frame, pip_px, (draw_x, draw_y), (0, 230, 60) if reciente else (255, 200, 0), 2, cv2.LINE_AA)
                        cv2.circle(frame, pip_px, 4, (0, 215, 255), -1, cv2.LINE_AA)
                        cv2.circle(frame, (draw_x, draw_y), 8, color_circulo, -1, cv2.LINE_AA)
                        cv2.circle(frame, (draw_x, draw_y), 8, (255, 255, 255), 1, cv2.LINE_AA)

                        # Etiqueta con formato de piano (Do, Re, Mi, Fa, Sol, La, Si)
                        cv2.putText(
                            frame, nombre_nota,
                            (draw_x - 14, draw_y + 24),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.58,
                            (0, 255, 80) if reciente else (255, 255, 255), 2, cv2.LINE_AA
                        )
            else:
                for i in range(10):
                    reset_finger(i)

            # Dibujar siempre el pentagrama en la parte superior derecha
            dibujar_pentagrama(frame, ahora)
            # Dibujar el botón "Salir" en la parte inferior derecha
            dibujar_boton_salir(frame)

        cv2.imshow(nombre_ventana, frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
        elif key in (13, 32) and estado_ui["en_menu"]:  # ENTER o ESPACIO también inician
            estado_ui["en_menu"] = False
        elif key in (ord('+'), ord('=')):
            UMBRAL_MOV_REL = max(0.006, UMBRAL_MOV_REL - 0.002)
        elif key in (ord('-'), ord('_')):
            UMBRAL_MOV_REL = min(0.040, UMBRAL_MOV_REL + 0.002)

    cap.release()
    cv2.destroyAllWindows()
    pygame.mixer.quit()

if __name__ == "__main__":
    main()
