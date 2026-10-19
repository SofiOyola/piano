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
archivos_por_dedo = ['C4', 'D4', 'E4', 'F4', 'G4', 'A4', 'B4', 'C5', 'D5', 'E5']
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
    min_detection_confidence=0.65,
    min_tracking_confidence=0.65
)

# Índices anatómicos fijos de las 5 puntas (NUNCA se reordenan por X cuadro a cuadro
# para evitar que dos dedos cercanos intercambien identidad y se aloquen las notas)
FINGER_TIPS = [4, 8, 12, 16, 20]   # Pulgar, Índice, Medio, Anular, Meñique
PALM_LANDMARKS = [0, 5, 9, 13, 17] # Muñeca y nudillos base (para medir quietud de la mano)

# --------------------------
# PARÁMETROS DE DETECCIÓN Y MODO STANDBY (MANO ACOSTADA / DERECHA)
# --------------------------
alpha_smooth = 0.55            # Suavizado EMA para eliminar temblores en postura horizontal
UMBRAL_VEL_DEDO = 0.024        # Velocidad de bajada en ventana corta (3 frames)
UMBRAL_BAJADA_STANDBY = 0.038  # Distancia que debe bajar el dedo desde su posición de Standby
MAX_MOV_PALMA = 0.014          # Si la palma/mano entera se mueve más que esto, queda en Standby
COOLDOWN = 0.28                # Tiempo mínimo entre toques del mismo dedo (segundos)
DURACION_NOTA_VISUAL = 0.70    # Tiempo que permanece iluminado el círculo en el pentagrama

# Estado por dedo (10 dedos)
prev_y = [None] * 10
prev_x = [None] * 10
baseline_y = [None] * 10       # Altura de reposo (Standby) de cada dedo
hist_y = [deque(maxlen=4) for _ in range(10)]
dedo_presionado = [False] * 10
frames_estables = [0] * 10     # Cuántos frames lleva el dedo estable tras aparecer o mover la mano
last_play = {i: 0.0 for i in range(10)}
ultimo_trigger_visual = {i: 0.0 for i in range(10)}

# Estado por mano (0: mano principal/izq, 1: mano secundaria/der)
prev_palm_pos = [None, None]
orden_invertido_mano = [None, None]  # Bloquea el orden de los 5 dedos para que no salten

def reset_finger(idx):
    prev_y[idx] = None
    prev_x[idx] = None
    baseline_y[idx] = None
    hist_y[idx].clear()
    dedo_presionado[idx] = False
    frames_estables[idx] = 0

def reset_hand(mano_slot):
    prev_palm_pos[mano_slot] = None
    orden_invertido_mano[mano_slot] = None
    for i in range(mano_slot * 5, mano_slot * 5 + 5):
        reset_finger(i)

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
# EXTRACCIÓN DE DEDOS CON IDENTIDAD ANATÓMICA FIJA (SIN INTERCAMBIOS)
# --------------------------
def extraer_mano_estable(lm, mano_slot, frame_width, frame_height):
    """
    Extrae las 5 puntas usando sus IDs anatómicos fijos [4, 8, 12, 16, 20].
    A diferencia del ordenamiento por X cuadro a cuadro, cada nota permanece atada
    estrictamente a su dedo físico para que al acostar la mano las notas NUNCA
    intercambien lugar ni se disparen solas.
    """
    # Centro de la palma (muñeca + nudillos MCP) para saber si toda la mano se está moviendo
    palm_x = float(np.mean([lm.landmark[p].x for p in PALM_LANDMARKS]))
    palm_y = float(np.mean([lm.landmark[p].y for p in PALM_LANDMARKS]))

    # Determinar una sola vez al entrar la mano si el pulgar (4) está a la derecha del meñique (20)
    # para asignar Do->Re->Mi->Fa->Sol de izquierda a derecha sin cambiarlo mientras se mueve.
    thumb_x = lm.landmark[4].x
    pinky_x = lm.landmark[20].x
    if orden_invertido_mano[mano_slot] is None:
        orden_invertido_mano[mano_slot] = (thumb_x > pinky_x)
    else:
        # Solo actualizar si hay una diferencia amplia y clara (evita parpadeos cuando está acostada)
        if (thumb_x - pinky_x) > 0.12:
            orden_invertido_mano[mano_slot] = True
        elif (pinky_x - thumb_x) > 0.12:
            orden_invertido_mano[mano_slot] = False

    tips_order = list(reversed(FINGER_TIPS)) if orden_invertido_mano[mano_slot] else list(FINGER_TIPS)

    dedos = []
    for tip_id in tips_order:
        x_norm = lm.landmark[tip_id].x
        y_norm = lm.landmark[tip_id].y
        dedos.append({
            "x_norm": x_norm,
            "y_norm": y_norm,
            "x_px": int(x_norm * frame_width),
            "y_px": int(y_norm * frame_height),
        })

    return dedos, (palm_x, palm_y)

def obtener_manos_estables(results, frame_width, frame_height):
    if not results.multi_hand_landmarks:
        return [None] * 10, [None, None]

    manos_raw = []
    for lm in results.multi_hand_landmarks:
        cx = float(np.mean([lm.landmark[p].x for p in PALM_LANDMARKS]))
        manos_raw.append((cx, lm))

    manos_raw.sort(key=lambda item: item[0])

    salida = [None] * 10
    palmas = [None, None]

    if len(manos_raw) >= 1:
        dedos_0, palma_0 = extraer_mano_estable(manos_raw[0][1], 0, frame_width, frame_height)
        for j in range(5):
            salida[j] = dedos_0[j]
        palmas[0] = palma_0
    else:
        reset_hand(0)

    if len(manos_raw) >= 2:
        dedos_1, palma_1 = extraer_mano_estable(manos_raw[1][1], 1, frame_width, frame_height)
        for j in range(5):
            salida[j + 5] = dedos_1[j]
        palmas[1] = palma_1
    else:
        reset_hand(1)

    return salida, palmas

# --------------------------
# PENTAGRAMA EN LA ESQUINA SUPERIOR DERECHA
# --------------------------
def dibujar_pentagrama(frame, ahora):
    """
    Dibuja el pentagrama en la parte superior derecha con un círculo en la nota
    que se está tocando en ese momento.
    """
    h, w, _ = frame.shape

    box_w = min(330, w - 20)
    box_h = 145
    x1 = w - box_w - 12
    y1 = 12
    x2 = w - 12
    y2 = y1 + box_h

    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (248, 246, 238), -1)
    cv2.addWeighted(overlay, 0.88, frame, 0.12, 0, frame)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (60, 60, 60), 2)

    notas_activas_nombres = [
        nombres_piano_por_dedo[i]
        for i in range(10)
        if (ahora - ultimo_trigger_visual.get(i, 0.0)) < DURACION_NOTA_VISUAL
    ]
    texto_titulo = "Pentagrama"
    if notas_activas_nombres:
        texto_titulo += f" - Tocando: {', '.join(notas_activas_nombres)}"
    else:
        texto_titulo += " (Standby)"

    cv2.putText(
        frame, texto_titulo, (x1 + 10, y1 + 20),
        cv2.FONT_HERSHEY_SIMPLEX, 0.46, (30, 30, 30), 1, cv2.LINE_AA
    )

    staff_left = x1 + 42
    staff_right = x2 - 14
    top_line_y = y1 + 38
    line_spacing = 14
    half_step = line_spacing // 2

    # 5 líneas del pentagrama
    for l_idx in range(5):
        ly = top_line_y + l_idx * line_spacing
        cv2.line(frame, (staff_left - 26, ly), (staff_right, ly), (50, 50, 50), 1, cv2.LINE_AA)

    bottom_line_y = top_line_y + 4 * line_spacing
    cv2.line(frame, (staff_left - 26, top_line_y), (staff_left - 26, bottom_line_y), (40, 40, 40), 2)
    cv2.line(frame, (staff_right, top_line_y), (staff_right, bottom_line_y), (40, 40, 40), 2)

    # Clave de Sol estilizada
    clef_x = staff_left - 14
    sol_line_y = top_line_y + 3 * line_spacing
    cv2.circle(frame, (clef_x, sol_line_y), 6, (40, 40, 40), 1, cv2.LINE_AA)
    cv2.line(frame, (clef_x + 3, top_line_y - 6), (clef_x - 2, bottom_line_y + 10), (40, 40, 40), 2, cv2.LINE_AA)
    cv2.circle(frame, (clef_x - 4, bottom_line_y + 10), 3, (40, 40, 40), -1, cv2.LINE_AA)

    pasos_por_dedo = [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
    ancho_util = staff_right - staff_left - 16

    for i in range(10):
        paso = pasos_por_dedo[i]
        nx = int(staff_left + 8 + (i * ancho_util / 9.0))
        ny = int(top_line_y + paso * half_step)
        nombre_nota = nombres_piano_por_dedo[i]
        activa = (ahora - ultimo_trigger_visual.get(i, 0.0)) < DURACION_NOTA_VISUAL

        if not activa:
            cv2.circle(frame, (nx, ny), 3, (175, 175, 175), -1, cv2.LINE_AA)
            cv2.putText(
                frame, nombre_nota, (nx - 8, y2 - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.34, (130, 130, 130), 1, cv2.LINE_AA
            )
        else:
            if i == 0:
                cv2.line(frame, (nx - 12, ny), (nx + 12, ny), (30, 30, 30), 2, cv2.LINE_AA)

            cv2.circle(frame, (nx, ny), 8, (0, 210, 70), -1, cv2.LINE_AA)
            cv2.circle(frame, (nx, ny), 8, (15, 70, 20), 2, cv2.LINE_AA)

            if paso >= 4:
                cv2.line(frame, (nx + 7, ny), (nx + 7, ny - 24), (15, 70, 20), 2, cv2.LINE_AA)
            else:
                cv2.line(frame, (nx - 7, ny), (nx - 7, ny + 24), (15, 70, 20), 2, cv2.LINE_AA)

            cv2.putText(
                frame, nombre_nota, (nx - 10, y2 - 7),
                cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0, 150, 40), 2, cv2.LINE_AA
            )

# --------------------------
# BUCLE PRINCIPAL
# --------------------------
def main():
    global UMBRAL_BAJADA_STANDBY, UMBRAL_VEL_DEDO

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("No se puede abrir la cámara.")
        return

    print("\n🎹 Piano con las Manos Activado — Modo Standby Inteligente")
    print("• Si acuestas la mano, las notas quedan en Standby (sin dispararse solas).")
    print("• Baja un dedo claramente hacia abajo para tocar esa nota.")
    print("• Controles: [+] más sensible | [-] más firme (requiere bajar más el dedo) | [ESC] salir\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error leyendo frame.")
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)
        ahora = time.time()

        if results.multi_hand_landmarks:
            tips, palmas = obtener_manos_estables(results, w, h)

            for mano_slot in (0, 1):
                palma_actual = palmas[mano_slot]
                if palma_actual is None:
                    continue

                # Medir cuánto se movió la mano entera (muñeca + nudillos)
                mov_palma = 0.0
                dy_palma = 0.0
                if prev_palm_pos[mano_slot] is not None:
                    dx_p = palma_actual[0] - prev_palm_pos[mano_slot][0]
                    dy_p = palma_actual[1] - prev_palm_pos[mano_slot][1]
                    mov_palma = float(np.hypot(dx_p, dy_p))
                    dy_palma = dy_p
                prev_palm_pos[mano_slot] = palma_actual

                # Si el usuario está acomodando, rotando o acostando la mano entera,
                # ponemos la mano en Standby y recalibramos la posición base de los dedos
                mano_en_traslado = (mov_palma > MAX_MOV_PALMA)

                indices_mano = range(mano_slot * 5, mano_slot * 5 + 5)
                datos_dedos = {}

                for i in indices_mano:
                    info = tips[i]
                    if info is None:
                        reset_finger(i)
                        continue

                    y_norm = info["y_norm"]
                    x_norm = info["x_norm"]

                    if prev_y[i] is None:
                        y_smooth = y_norm
                        x_smooth = x_norm
                        baseline_y[i] = y_norm
                        frames_estables[i] = 0
                    else:
                        # Zona muerta pequeña para ignorar vibraciones de cámara en Standby
                        if abs(y_norm - prev_y[i]) < 0.003:
                            y_smooth = prev_y[i]
                        else:
                            y_smooth = alpha_smooth * y_norm + (1.0 - alpha_smooth) * prev_y[i]
                        x_smooth = alpha_smooth * x_norm + (1.0 - alpha_smooth) * prev_x[i]

                    hist_y[i].append(y_smooth)

                    if mano_en_traslado:
                        # Mientras se acuesta o traslada la mano, actualizar Standby sin tocar
                        baseline_y[i] = y_smooth
                        frames_estables[i] = 0
                    else:
                        frames_estables[i] += 1

                    # Movimiento en ventana corta (3 frames) descontando el movimiento vertical de la palma
                    dy_ventana = (y_smooth - hist_y[i][0]) if len(hist_y[i]) >= 2 else 0.0
                    dy_neto = dy_ventana - max(0.0, dy_palma * len(hist_y[i]) * 0.85)

                    # Si el dedo está quieto o subiendo, la línea de Standby lo sigue suavemente
                    # para estar lista esperando a que el usuario baje el dedo
                    if not dedo_presionado[i] and dy_neto < (UMBRAL_VEL_DEDO * 0.45):
                        baseline_y[i] = 0.88 * baseline_y[i] + 0.12 * y_smooth
                    elif y_smooth < baseline_y[i]:
                        # Si el dedo sube más arriba de su nivel de Standby, actualizar el techo de Standby
                        baseline_y[i] = y_smooth

                    bajada_desde_standby = y_smooth - baseline_y[i]
                    datos_dedos[i] = (dy_neto, bajada_desde_standby, int(x_smooth * w), int(y_smooth * h))

                    prev_y[i] = y_smooth
                    prev_x[i] = x_smooth

                if not datos_dedos:
                    continue

                # Encontrar cuál es el dedo que más está bajando en esta mano
                # (cuando la mano está acostada, evita que el temblor de dedos vecinos dispare notas)
                max_bajada_mano = max(v[1] for v in datos_dedos.values())

                for i, (dy_neto, bajada_desde_standby, draw_x, draw_y) in datos_dedos.items():
                    nombre_nota = nombres_piano_por_dedo[i]

                    if not dedo_presionado[i]:
                        # Solo puede tocar si la mano ya se estabilizó en Standby (>= 4 frames quieta)
                        # y el dedo bajó bastante desde su punto de Standby (bajada_desde_standby)
                        # con velocidad descendente clara (dy_neto).
                        es_dedo_dominante = (bajada_desde_standby >= max_bajada_mano * 0.75)
                        if (
                            not mano_en_traslado
                            and frames_estables[i] >= 4
                            and es_dedo_dominante
                            and bajada_desde_standby > UMBRAL_BAJADA_STANDBY
                            and dy_neto > UMBRAL_VEL_DEDO
                        ):
                            if tocar_sonido_idx(i):
                                dedo_presionado[i] = True
                    else:
                        # Para volver a estar en Standby y poder tocar otra vez,
                        # el dedo debe subir de regreso hacia su posición de espera
                        if bajada_desde_standby < (UMBRAL_BAJADA_STANDBY * 0.40) or dy_neto < -0.008:
                            dedo_presionado[i] = False
                            baseline_y[i] = prev_y[i]

                    # Dibujar únicamente los 5 puntos limpios en las puntas y su nota de piano
                    reciente = (ahora - ultimo_trigger_visual.get(i, 0.0)) < 0.30
                    color_circulo = (0, 230, 60) if reciente else (255, 90, 0)

                    cv2.circle(frame, (draw_x, draw_y), 8, color_circulo, -1, cv2.LINE_AA)
                    cv2.circle(frame, (draw_x, draw_y), 8, (255, 255, 255), 1, cv2.LINE_AA)

                    cv2.putText(
                        frame, nombre_nota,
                        (draw_x - 14, draw_y + 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.58,
                        (0, 255, 80) if reciente else (255, 255, 255), 2, cv2.LINE_AA
                    )
        else:
            reset_hand(0)
            reset_hand(1)

        # Pentagrama arriba a la derecha
        dibujar_pentagrama(frame, ahora)

        cv2.imshow("Piano Virtual - Mano en Standby + Pentagrama", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
        elif key in (ord('+'), ord('=')):
            # Más sensible (requiere bajar menos el dedo)
            UMBRAL_BAJADA_STANDBY = max(0.018, UMBRAL_BAJADA_STANDBY - 0.004)
            UMBRAL_VEL_DEDO = max(0.012, UMBRAL_VEL_DEDO - 0.002)
        elif key in (ord('-'), ord('_')):
            # Más firme (requiere bajar más el dedo desde Standby)
            UMBRAL_BAJADA_STANDBY = min(0.075, UMBRAL_BAJADA_STANDBY + 0.004)
            UMBRAL_VEL_DEDO = min(0.045, UMBRAL_VEL_DEDO + 0.002)

    cap.release()
    cv2.destroyAllWindows()
    pygame.mixer.quit()

if __name__ == "__main__":
    main()
