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
# MEDIAPIPE HANDS (Umbral de tracking bajo para NO perder la mano al acostarla)
# --------------------------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    model_complexity=1,
    min_detection_confidence=0.35,
    min_tracking_confidence=0.15
)

FINGER_TIPS = [4, 8, 12, 16, 20]
PALM_LANDMARKS = [0, 5, 9, 13, 17]

# --------------------------
# ESTADO Y DETECCIÓN DUAL: MANO DERECHA vs MANO ACOSTADA (STANDBY)
# --------------------------
# IMPORTANTE: El historial (prev_y, baseline_y) se guarda por (mano_slot, tip_id_anatomico)
# y NUNCA por orden X cambiante, evitando que dos dedos cercanos intercambien su altura Y
# y disparen notas falsas cuando la mano está acostada.

UMBRAL_UPRIGHT = 0.018          # Sensibilidad cuando la mano está derecha (vertical)
UMBRAL_ACOSTADA_BAJADA = 0.045  # Bajada profunda requerida cuando la mano está acostada en Standby
UMBRAL_ACOSTADA_DIF = 0.032     # Cuánto más debe bajar ese dedo respecto a los otros 4 dedos en Standby
COOLDOWN = 0.25                 # Segundos mínimos entre toques de la misma nota
DURACION_NOTA_VISUAL = 0.70     # Segundos que se ilumina la nota en el pentagrama

# Diccionarios indexados por clave física: (mano_slot, tip_id) donde tip_id in [4, 8, 12, 16, 20]
prev_x_phys = {}
prev_y_phys = {}
baseline_y_phys = {}
hist_y_phys = {}
dedo_abajo_phys = {}

# Bloqueo temporal de Standby cuando el usuario está acostando o moviendo la mano entera
standby_bloqueo_hasta = [0.0, 0.0]
prev_palm_pos = [None, None]
prev_hand_span_y = [None, None]
orden_x_estable = [None, None]  # Orden de los 5 tip_ids de izquierda a derecha con histéresis

# Seguimiento por flujo óptico (Optical Flow) para que NO desaparezcan los puntos
# si MediaPipe pierde momentáneamente la palma cuando la mano está 100% horizontal
prev_gray = None
ultimo_lm_conocido = [None, None]
frames_sin_mp = [999, 999]
MAX_FRAMES_OPTICAL_FLOW = 45    # Mantiene la mano visible hasta ~1.5s si MediaPipe falla en horizontal

last_play = {i: 0.0 for i in range(10)}
ultimo_trigger_visual = {i: 0.0 for i in range(10)}

def reset_mano(m_slot):
    for tid in FINGER_TIPS:
        key = (m_slot, tid)
        prev_x_phys.pop(key, None)
        prev_y_phys.pop(key, None)
        baseline_y_phys.pop(key, None)
        hist_y_phys.pop(key, None)
        dedo_abajo_phys.pop(key, None)
    prev_palm_pos[m_slot] = None
    prev_hand_span_y[m_slot] = None
    orden_x_estable[m_slot] = None
    ultimo_lm_conocido[m_slot] = None
    frames_sin_mp[m_slot] = 999

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

def obtener_orden_estable_x(m_slot, coords_por_tip):
    """
    Ordena los 5 tip_ids de izquierda a derecha en X, pero con histéresis fuerte (3.5% del ancho)
    para que cuando la mano esté acostada dos dedos cercanos NUNCA intercambien su nota asignada.
    """
    actual = orden_x_estable[m_slot]
    if actual is None:
        orden_x_estable[m_slot] = sorted(FINGER_TIPS, key=lambda tid: coords_por_tip[tid][0])
        return orden_x_estable[m_slot]

    # Solo permitir intercambio entre dos dedos adyacentes si realmente se cruzaron por > 0.035
    lista = list(actual)
    for k in range(4):
        t_izq = lista[k]
        t_der = lista[k + 1]
        if coords_por_tip[t_izq][0] - coords_por_tip[t_der][0] > 0.035:
            lista[k], lista[k + 1] = t_der, t_izq
    orden_x_estable[m_slot] = lista
    return lista

# --------------------------
# PENTAGRAMA EN LA ESQUINA SUPERIOR DERECHA
# --------------------------
def dibujar_pentagrama(frame, ahora, en_modo_acostada):
    """
    Dibuja el pentagrama musical en la parte superior derecha con un círculo verde
    en la nota que se está tocando en ese momento.
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
    if notas_activas_nombres:
        texto_titulo = f"Pentagrama - Tocando: {', '.join(notas_activas_nombres)}"
    elif en_modo_acostada:
        texto_titulo = "Pentagrama (Mano acostada: Standby)"
    else:
        texto_titulo = "Pentagrama"

    cv2.putText(
        frame, texto_titulo, (x1 + 10, y1 + 20),
        cv2.FONT_HERSHEY_SIMPLEX, 0.44, (30, 30, 30), 1, cv2.LINE_AA
    )

    staff_left = x1 + 42
    staff_right = x2 - 14
    top_line_y = y1 + 38
    line_spacing = 14
    half_step = line_spacing // 2

    # 5 líneas horizontales del pentagrama
    for l_idx in range(5):
        ly = top_line_y + l_idx * line_spacing
        cv2.line(frame, (staff_left - 26, ly), (staff_right, ly), (50, 50, 50), 1, cv2.LINE_AA)

    bottom_line_y = top_line_y + 4 * line_spacing
    cv2.line(frame, (staff_left - 26, top_line_y), (staff_left - 26, bottom_line_y), (40, 40, 40), 2)
    cv2.line(frame, (staff_right, top_line_y), (staff_right, bottom_line_y), (40, 40, 40), 2)

    # Símbolo estilizado de Clave de Sol
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
    global UMBRAL_UPRIGHT, UMBRAL_ACOSTADA_BAJADA, prev_gray

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("No se puede abrir la cámara.")
        return

    print("\n🎹 Piano con las Manos Activado")
    print("• Mano derecha (vertical): toque normal sensible.")
    print("• Mano acostada (horizontal): queda en Standby sin alocarse; baja bastante un dedo para tocar.")
    print("• Controles: [+] más sensible | [-] más firme | [ESC] salir\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error leyendo frame.")
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = hands.process(rgb)
        ahora = time.time()

        # Extraer manos de MediaPipe o mantener con Optical Flow si la mano está horizontal
        manos_actuales = [None, None]
        if results.multi_hand_landmarks:
            lista_mp = []
            for lm in results.multi_hand_landmarks:
                cx = float(np.mean([lm.landmark[p].x for p in PALM_LANDMARKS]))
                pts = {idx: (lm.landmark[idx].x, lm.landmark[idx].y) for idx in (FINGER_TIPS + PALM_LANDMARKS)}
                lista_mp.append((cx, pts))
            lista_mp.sort(key=lambda item: item[0])

            for m_slot in range(min(2, len(lista_mp))):
                manos_actuales[m_slot] = lista_mp[m_slot][1]
                ultimo_lm_conocido[m_slot] = lista_mp[m_slot][1]
                frames_sin_mp[m_slot] = 0

            if len(lista_mp) < 2:
                reset_mano(1)
        else:
            # Si MediaPipe pierde la mano porque se puso 100% horizontal de canto,
            # la seguimos con Optical Flow (Lucas-Kanade) para que NO desaparezcan las notas
            for m_slot in (0, 1):
                if ultimo_lm_conocido[m_slot] is not None and prev_gray is not None and frames_sin_mp[m_slot] < MAX_FRAMES_OPTICAL_FLOW:
                    pts_prev = ultimo_lm_conocido[m_slot]
                    keys_pts = list(pts_prev.keys())
                    p0 = np.array([[[pts_prev[k][0] * w, pts_prev[k][1] * h]] for k in keys_pts], dtype=np.float32)
                    p1, st, _ = cv2.calcOpticalFlowPyrLK(
                        prev_gray, gray, p0, None,
                        winSize=(21, 21), maxLevel=3,
                        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 15, 0.03)
                    )
                    if p1 is not None and st is not None and np.sum(st) >= 6:
                        nuevos_pts = {}
                        for idx_k, k in enumerate(keys_pts):
                            nx_norm = float(np.clip(p1[idx_k, 0, 0] / w, 0.02, 0.98))
                            ny_norm = float(np.clip(p1[idx_k, 0, 1] / h, 0.02, 0.98))
                            nuevos_pts[k] = (nx_norm, ny_norm)
                        manos_actuales[m_slot] = nuevos_pts
                        ultimo_lm_conocido[m_slot] = nuevos_pts
                        frames_sin_mp[m_slot] += 1
                    else:
                        reset_mano(m_slot)
                else:
                    reset_mano(m_slot)

        prev_gray = gray
        alguna_mano_acostada = False

        for m_slot in (0, 1):
            pts = manos_actuales[m_slot]
            if pts is None:
                continue

            # 1. Detectar si la mano está DERECHA (vertical) o ACOSTADA (horizontal)
            all_ys = [pts[k][1] for k in pts]
            all_xs = [pts[k][0] for k in pts]
            span_y = max(all_ys) - min(all_ys)
            span_x = max(all_xs) - min(all_xs)

            # En una mano acostada (como en tu foto), la altura vertical (span_y) es pequeña
            # en comparación con el ancho horizontal (span_x) o menor a 0.22
            es_mano_acostada = (span_y < 0.22) or (span_x > span_y * 1.15)
            if es_mano_acostada:
                alguna_mano_acostada = True

            # 2. Detectar si el usuario está moviendo/acostando toda la mano -> Activar Standby
            palm_x = float(np.mean([pts[p][0] for p in PALM_LANDMARKS]))
            palm_y = float(np.mean([pts[p][1] for p in PALM_LANDMARKS]))

            mov_palma = 0.0
            dy_palma = 0.0
            if prev_palm_pos[m_slot] is not None:
                dx_p = palm_x - prev_palm_pos[m_slot][0]
                dy_p = palm_y - prev_palm_pos[m_slot][1]
                mov_palma = float(np.hypot(dx_p, dy_p))
                dy_palma = dy_p
            prev_palm_pos[m_slot] = (palm_x, palm_y)

            cambio_span = 0.0
            if prev_hand_span_y[m_slot] is not None:
                cambio_span = abs(span_y - prev_hand_span_y[m_slot])
            prev_hand_span_y[m_slot] = span_y

            # Si la mano se está acostando (cambia span_y) o trasladando rápido,
            # bloquear disparos por 0.30s para que quede en Standby total sin alocarse
            if mov_palma > 0.018 or cambio_span > 0.018:
                standby_bloqueo_hasta[m_slot] = max(standby_bloqueo_hasta[m_slot], ahora + 0.30)

            en_bloqueo_standby = (ahora < standby_bloqueo_hasta[m_slot])

            # 3. Actualizar posición suavizada de cada dedo físico (tip_id in [4, 8, 12, 16, 20])
            alpha = 0.42 if es_mano_acostada else 0.65
            deadzone = 0.005 if es_mano_acostada else 0.001

            coords_suavizadas = {}
            dy_instantaneo = {}
            bajada_standby = {}

            for tid in FINGER_TIPS:
                key = (m_slot, tid)
                raw_x, raw_y = pts[tid]

                if key not in prev_y_phys:
                    sx, sy = raw_x, raw_y
                    baseline_y_phys[key] = raw_y
                    hist_y_phys[key] = deque([raw_y], maxlen=4)
                    dedo_abajo_phys[key] = False
                else:
                    if abs(raw_y - prev_y_phys[key]) < deadzone:
                        sy = prev_y_phys[key]
                    else:
                        sy = alpha * raw_y + (1.0 - alpha) * prev_y_phys[key]
                    sx = alpha * raw_x + (1.0 - alpha) * prev_x_phys[key]

                dy_1 = sy - prev_y_phys.get(key, sy)
                hist_y_phys[key].append(sy)
                dy_ventana = sy - hist_y_phys[key][0]

                # Si estamos en bloqueo por estar acostando la mano, fijar el Standby al nivel actual
                if en_bloqueo_standby:
                    baseline_y_phys[key] = sy
                    dedo_abajo_phys[key] = False
                else:
                    # En modo normal, la referencia de Standby sigue lentamente al dedo cuando está en reposo
                    if not dedo_abajo_phys[key] and abs(dy_ventana) < 0.010:
                        baseline_y_phys[key] = 0.90 * baseline_y_phys[key] + 0.10 * sy
                    elif sy < baseline_y_phys[key]:
                        baseline_y_phys[key] = sy

                coords_suavizadas[tid] = (sx, sy)
                dy_instantaneo[tid] = dy_1 - max(0.0, dy_palma * 0.8)
                bajada_standby[tid] = sy - baseline_y_phys[key]

                prev_x_phys[key] = sx
                prev_y_phys[key] = sy

            # 4. Asignar notas de izquierda a derecha con histéresis para que no salten
            orden_tips = obtener_orden_estable_x(m_slot, coords_suavizadas)

            # Mediana de bajada de la mano (para saber si un dedo bajó mucho más que el resto)
            lista_bajadas = list(bajada_standby.values())
            mediana_bajada = float(np.median(lista_bajadas))
            max_bajada = max(lista_bajadas)

            for pos_idx, tid in enumerate(orden_tips):
                nota_idx = m_slot * 5 + pos_idx
                nombre_nota = nombres_piano_por_dedo[nota_idx]
                key = (m_slot, tid)
                sx, sy = coords_suavizadas[tid]
                draw_x, draw_y = int(sx * w), int(sy * h)

                if not en_bloqueo_standby:
                    if not es_mano_acostada:
                        # MODO MANO DERECHA (VERTICAL): comportamiento directo que te gustó
                        if not dedo_abajo_phys[key]:
                            if dy_instantaneo[tid] > UMBRAL_UPRIGHT:
                                tocar_sonido_idx(nota_idx)
                                dedo_abajo_phys[key] = True
                        else:
                            if dy_instantaneo[tid] < (UMBRAL_UPRIGHT * 0.3):
                                dedo_abajo_phys[key] = False
                    else:
                        # MODO MANO ACOSTADA (STANDBY):
                        # Queda quieta en Standby sin alocarse y SOLO toca si bajas un dedo MUCHO
                        bajada_dedo = bajada_standby[tid]
                        diferencia_vs_otros = bajada_dedo - max(0.0, mediana_bajada)

                        if not dedo_abajo_phys[key]:
                            if (
                                bajada_dedo == max_bajada
                                and bajada_dedo > UMBRAL_ACOSTADA_BAJADA
                                and diferencia_vs_otros > UMBRAL_ACOSTADA_DIF
                                and dy_instantaneo[tid] > 0.008
                            ):
                                if tocar_sonido_idx(nota_idx):
                                    dedo_abajo_phys[key] = True
                        else:
                            # Se rearma a Standby cuando subes el dedo
                            if bajada_dedo < (UMBRAL_ACOSTADA_BAJADA * 0.45):
                                dedo_abajo_phys[key] = False
                                baseline_y_phys[key] = sy

                # Dibujar el punto y el nombre de la nota (Do, Re, Mi, Fa, Sol...)
                reciente = (ahora - ultimo_trigger_visual.get(nota_idx, 0.0)) < 0.30
                color_circulo = (0, 230, 60) if reciente else (255, 90, 0)

                cv2.circle(frame, (draw_x, draw_y), 8, color_circulo, -1, cv2.LINE_AA)
                cv2.circle(frame, (draw_x, draw_y), 8, (255, 255, 255), 1, cv2.LINE_AA)

                cv2.putText(
                    frame, nombre_nota,
                    (draw_x - 14, draw_y + 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.58,
                    (0, 255, 80) if reciente else (255, 255, 255), 2, cv2.LINE_AA
                )

        # Dibujar el pentagrama arriba a la derecha
        dibujar_pentagrama(frame, ahora, alguna_mano_acostada)

        cv2.imshow("Piano Virtual - Mano Derecha y Mano Acostada (Standby)", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
        elif key in (ord('+'), ord('=')):
            UMBRAL_UPRIGHT = max(0.008, UMBRAL_UPRIGHT - 0.002)
            UMBRAL_ACOSTADA_BAJADA = max(0.025, UMBRAL_ACOSTADA_BAJADA - 0.004)
        elif key in (ord('-'), ord('_')):
            UMBRAL_UPRIGHT = min(0.040, UMBRAL_UPRIGHT + 0.002)
            UMBRAL_ACOSTADA_BAJADA = min(0.080, UMBRAL_ACOSTADA_BAJADA + 0.004)

    cap.release()
    cv2.destroyAllWindows()
    pygame.mixer.quit()

if __name__ == "__main__":
    main()
