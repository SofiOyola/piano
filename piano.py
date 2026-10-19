import os
import sys
import warnings

# Evitar conflicto entre TensorFlow y Protobuf al importar MediaPipe
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
# DETECTORES DE PUNTAS DE DEDO (Mano Derecha + Mano Acostada Directa)
# --------------------------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    model_complexity=1,
    min_detection_confidence=0.50,
    min_tracking_confidence=0.50
)

# Segmentador de primer plano y detector facial para encontrar las 5 puntas directamente
# cuando la mano está acostada horizontalmente (donde MediaPipe Hands no ve la palma)
mp_selfie = mp.solutions.selfie_segmentation
segmenter = mp_selfie.SelfieSegmentation(model_selection=1)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Usar EXCLUSIVAMENTE las 5 puntas
FINGER_TIPS = [4, 8, 12, 16, 20]

# --------------------------
# PARÁMETROS DE SEGUIMIENTO EXCLUSIVO DE PUNTAS
# --------------------------
UMBRAL_BAJADA_VERTICAL = 0.018    # Sensibilidad cuando la mano está derecha
UMBRAL_BAJADA_ACOSTADA = 0.028    # Bajada necesaria del dedo cuando la mano está acostada en Standby
COOLDOWN = 0.24                   # Segundos entre repeticiones del mismo dedo
DURACION_NOTA_VISUAL = 0.70       # Segundos que permanece iluminado el círculo en el pentagrama

# Estado de las 10 puntas de los dedos
puntas_xy = [None] * 10           # Posición actual suavizada (x_px, y_px) de cada punta
baseline_rel_y = [None] * 10      # Posición relativa de Standby de cada punta respecto al grupo de 5 puntas
hist_flow_dy = [deque(maxlen=4) for _ in range(10)]
dedo_presionado = [False] * 10
frames_activos = [0] * 10
last_play = {i: 0.0 for i in range(10)}
ultimo_trigger_visual = {i: 0.0 for i in range(10)}

prev_gray = None
ultima_cara = None
frames_desde_cara = 999

def reset_dedo(idx):
    puntas_xy[idx] = None
    baseline_rel_y[idx] = None
    hist_flow_dy[idx].clear()
    dedo_presionado[idx] = False
    frames_activos[idx] = 0

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
# DETECTOR DIRECTO DE LAS 5 PUNTAS DE LOS DEDOS (CUALQUIER POSTURA)
# --------------------------
def detectar_puntas_mano_acostada(bgr_frame, rgb_frame, gray_frame):
    """
    Detecta directamente las 5 puntas de los dedos cuando la mano está acostada
    horizontalmente frente a la cámara (donde MediaPipe Hands falla o confunde la palma).
    Retorna lista de listas de 5 tuplas [(x_px, y_px), ...] ordenadas de izquierda a derecha.
    """
    global ultima_cara, frames_desde_cara
    h, w, _ = bgr_frame.shape

    # 1. Máscara de la persona en primer plano (elimina cortinas y fondo)
    res_seg = segmenter.process(rgb_frame)
    if res_seg.segmentation_mask is None:
        return []
    person_mask = (res_seg.segmentation_mask > 0.22).astype(np.uint8) * 255
    person_mask = cv2.dilate(person_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13)))

    # 2. Máscara de piel (R > G > B y rango YCrCb para excluir ropa rosada/oscura)
    b, g, r = cv2.split(bgr_frame)
    ycrcb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2YCrCb)
    cr = ycrcb[:, :, 1]
    cb = ycrcb[:, :, 2]

    skin_cond = (
        (r.astype(np.int16) - g.astype(np.int16) > 6) &
        (g.astype(np.int16) - b.astype(np.int16) > 3) &
        (r > 55) & (r < 210) &
        (cr >= 130) & (cr <= 172) &
        (cb >= 77) & (cb <= 132)
    ).astype(np.uint8) * 255

    hand_skin = cv2.bitwise_and(skin_cond, person_mask)

    # 3. Excluir el rostro y el cuello para quedarnos únicamente con la(s) mano(s)
    faces = face_cascade.detectMultiScale(gray_frame, 1.15, 4, minSize=(60, 60))
    if len(faces) > 0:
        # Tomar el rostro principal
        faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
        ultima_cara = faces[0]
        frames_desde_cara = 0
    else:
        frames_desde_cara += 1

    if ultima_cara is not None and frames_desde_cara < 30:
        fx, fy, fw, fh = ultima_cara
        cv2.rectangle(
            hand_skin,
            (max(0, fx - int(fw * 0.18)), 0),
            (min(w, fx + fw + int(fw * 0.22)), min(h, fy + int(fh * 1.42))),
            0, -1
        )

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    hand_skin = cv2.morphologyEx(hand_skin, cv2.MORPH_OPEN, kernel)
    hand_skin = cv2.morphologyEx(hand_skin, cv2.MORPH_CLOSE, kernel)

    cnts, _ = cv2.findContours(hand_skin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = [c for c in cnts if cv2.contourArea(c) > (w * h * 0.018)]
    cnts.sort(key=cv2.contourArea, reverse=True)

    manos_puntas = []
    rel_xs = [0.07, 0.28, 0.52, 0.74, 0.93]
    gray_blur = cv2.GaussianBlur(gray_frame, (5, 5), 0)

    for c in cnts[:2]:
        x, y, cw, ch = cv2.boundingRect(c)
        if cw < int(w * 0.16):
            continue

        hand_only = np.zeros_like(hand_skin)
        cv2.drawContours(hand_only, [c], -1, 255, -1)

        # Banda superior donde están las 5 puntas de la mano acostada
        alto_banda = min(int(h * 0.15), 72)
        upper_band = hand_only[y:min(h, y + alto_banda), :]
        cols = np.where(np.sum(upper_band > 0, axis=0) > 4)[0]
        if len(cols) < 30:
            continue

        lx, rx = int(cols[0]), int(cols[-1])
        if (rx - lx) < int(w * 0.15):
            continue

        puntas_5 = []
        for k in range(5):
            cx = int(lx + rel_xs[k] * (rx - lx))
            y_start = max(0, y + 8)
            y_end = min(h - 4, y + alto_banda + 6)
            x_l = max(0, cx - 7)
            x_r = min(w, cx + 8)

            patch_gray = gray_blur[y_start:y_end, x_l:x_r]
            patch_mask = hand_only[y_start:y_end, x_l:x_r]

            if patch_gray.shape[0] > 6 and np.any(patch_mask > 0):
                profile = np.mean(patch_gray * (patch_mask > 0), axis=1)
                grad_y = np.gradient(profile)
                # La punta del dedo está justo encima de la sombra inferior del dedo
                edge_idx = int(np.argmin(grad_y[3:])) + 3
                ty = y_start + max(4, edge_idx - 4)
            else:
                ty = y + 32

            puntas_5.append((cx, ty))

        manos_puntas.append((lx, puntas_5))

    manos_puntas.sort(key=lambda item: item[0])
    return [item[1] for item in manos_puntas]

def obtener_solo_puntas(bgr_frame, rgb_frame, gray_frame):
    """
    Obtiene EXCLUSIVAMENTE las 5 puntas de cada mano (ordenadas de izquierda a derecha),
    sin importar si la mano está derecha (vertical) o acostada (horizontal).
    Retorna:
      - lista de hasta 2 manos, cada una con [(x_px, y_px) x 5, es_acostada]
    """
    h, w, _ = bgr_frame.shape
    results = hands.process(rgb_frame)

    manos_verticales = []
    if results.multi_hand_landmarks:
        for lm in results.multi_hand_landmarks:
            tips_y = [lm.landmark[t].y for t in FINGER_TIPS]
            tips_x = [lm.landmark[t].x for t in FINGER_TIPS]
            wrist_y = lm.landmark[0].y
            span_y = max(tips_y + [wrist_y]) - min(tips_y)
            span_x = max(tips_x) - min(tips_x)

            # Solo aceptamos MediaPipe Hands si la mano está claramente DERECHA (vertical):
            # las puntas están bien arriba de la muñeca y la mano no está aplastada horizontalmente.
            es_vertical_clara = ((wrist_y - min(tips_y)) > 0.20) and (span_y > span_x * 0.90)
            if es_vertical_clara:
                puntas = [(int(lm.landmark[t].x * w), int(lm.landmark[t].y * h)) for t in FINGER_TIPS]
                puntas.sort(key=lambda p: p[0])
                cx_mano = float(np.mean([p[0] for p in puntas]))
                manos_verticales.append((cx_mano, puntas, False))

    if manos_verticales:
        manos_verticales.sort(key=lambda item: item[0])
        return [(item[1], item[2]) for item in manos_verticales[:2]]

    # Si la mano NO está vertical (está acostada / en forma de araña),
    # detectamos directamente las 5 puntas de los dedos sin usar el esqueleto de palma
    manos_acostadas = detectar_puntas_mano_acostada(bgr_frame, rgb_frame, gray_frame)
    return [(puntas, True) for puntas in manos_acostadas[:2]]

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
        texto_titulo = "Pentagrama (Standby)"
    else:
        texto_titulo = "Pentagrama"

    cv2.putText(
        frame, texto_titulo, (x1 + 10, y1 + 20),
        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (30, 30, 30), 1, cv2.LINE_AA
    )

    staff_left = x1 + 42
    staff_right = x2 - 14
    top_line_y = y1 + 38
    line_spacing = 14
    half_step = line_spacing // 2

    for l_idx in range(5):
        ly = top_line_y + l_idx * line_spacing
        cv2.line(frame, (staff_left - 26, ly), (staff_right, ly), (50, 50, 50), 1, cv2.LINE_AA)

    bottom_line_y = top_line_y + 4 * line_spacing
    cv2.line(frame, (staff_left - 26, top_line_y), (staff_left - 26, bottom_line_y), (40, 40, 40), 2)
    cv2.line(frame, (staff_right, top_line_y), (staff_right, bottom_line_y), (40, 40, 40), 2)

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
    global UMBRAL_BAJADA_VERTICAL, UMBRAL_BAJADA_ACOSTADA, prev_gray

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("No se puede abrir la cámara.")
        return

    print("\n🎹 Piano con las Manos — Detección Exclusiva de Puntas de Dedo")
    print("• Detecta SOLO las puntas de los dedos (mano derecha o mano acostada).")
    print("• En mano acostada queda en Standby y toca cuando bajas un dedo.")
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
        ahora = time.time()

        manos_detectadas = obtener_solo_puntas(frame, rgb, gray)
        alguna_acostada = False

        # Limpiar manos no presentes
        for m_slot in range(len(manos_detectadas), 2):
            for idx in range(m_slot * 5, m_slot * 5 + 5):
                reset_dedo(idx)

        for m_slot, (puntas_det, es_acostada) in enumerate(manos_detectadas):
            if es_acostada:
                alguna_acostada = True

            indices_mano = list(range(m_slot * 5, m_slot * 5 + 5))
            flow_dys_mano = []

            # 1. Seguir el movimiento real en píxeles de cada punta con Optical Flow (Lucas-Kanade)
            # Esto garantiza que si el dedo físico NO se movió hacia abajo en el video,
            # la nota jamás se dispare sola en Standby.
            for j, i in enumerate(indices_mano):
                det_x, det_y = float(puntas_det[j][0]), float(puntas_det[j][1])

                if puntas_xy[i] is None or prev_gray is None:
                    puntas_xy[i] = (det_x, det_y)
                    hist_flow_dy[i].clear()
                    hist_flow_dy[i].append(0.0)
                    frames_activos[i] = 1
                    flow_dys_mano.append(0.0)
                    continue

                prev_x, prev_y = puntas_xy[i]
                p0 = np.array([[[prev_x, prev_y]]], dtype=np.float32)
                p1, st, _ = cv2.calcOpticalFlowPyrLK(
                    prev_gray, gray, p0, None,
                    winSize=(23, 23), maxLevel=3,
                    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 15, 0.03)
                )

                if p1 is not None and st is not None and st[0, 0] == 1:
                    fx, fy = float(p1[0, 0, 0]), float(p1[0, 0, 1])
                    dy_flow_norm = (fy - prev_y) / float(h)
                else:
                    fx, fy = det_x, det_y
                    dy_flow_norm = (det_y - prev_y) / float(h)

                # Combinar suavemente la posición guiada por flujo óptico con el detector de puntas
                alpha_pos = 0.28 if es_acostada else 0.65
                # Si en mano acostada el píxel casi no se movió en el video, amortiguar cualquier salto
                if es_acostada and abs(dy_flow_norm) < 0.003:
                    alpha_pos = 0.12

                nx = alpha_pos * det_x + (1.0 - alpha_pos) * fx
                ny = alpha_pos * det_y + (1.0 - alpha_pos) * fy

                # Desplazamiento vertical combinado (real en imagen + posición de punta)
                dy_efectivo_frame = 0.65 * dy_flow_norm + 0.35 * ((ny - prev_y) / float(h))
                hist_flow_dy[i].append(dy_efectivo_frame)
                puntas_xy[i] = (nx, ny)
                frames_activos[i] += 1
                flow_dys_mano.append(sum(hist_flow_dy[i]))

            # 2. Calcular la línea de referencia de las 5 puntas de la mano
            ys_norm = [puntas_xy[i][1] / float(h) for i in indices_mano]
            mediana_y_mano = float(np.median(ys_norm))
            mediana_flow_mano = float(np.median(flow_dys_mano)) if flow_dys_mano else 0.0

            # Si toda la mano se está moviendo/bajando en bloque, ninguna nota debe dispararse
            movimiento_bloque = abs(mediana_flow_mano) > 0.020

            # Evaluar la bajada relativa de cada punta respecto a su línea de Standby
            bajadas_actuales = {}
            for j, i in enumerate(indices_mano):
                rel_y = ys_norm[j] - mediana_y_mano
                flow_ventana = flow_dys_mano[j] - max(0.0, mediana_flow_mano)

                if baseline_rel_y[i] is None or movimiento_bloque or frames_activos[i] < 5:
                    baseline_rel_y[i] = rel_y
                    dedo_presionado[i] = False
                else:
                    # Mientras el dedo está en Standby (sin bajar rápido), adaptar suavemente su base
                    if not dedo_presionado[i] and abs(flow_ventana) < 0.008:
                        baseline_rel_y[i] = 0.90 * baseline_rel_y[i] + 0.10 * rel_y
                    elif rel_y < baseline_rel_y[i]:
                        baseline_rel_y[i] = rel_y

                bajada_relativa = rel_y - baseline_rel_y[i]
                bajadas_actuales[i] = (bajada_relativa, flow_ventana)

            # Encontrar el dedo con mayor bajada en la mano
            max_score_mano = max(
                (b_rel + f_win) for (b_rel, f_win) in bajadas_actuales.values()
            ) if bajadas_actuales else 0.0

            for j, i in enumerate(indices_mano):
                draw_x, draw_y = int(puntas_xy[i][0]), int(puntas_xy[i][1])
                nombre_nota = nombres_piano_por_dedo[i]
                bajada_relativa, flow_ventana = bajadas_actuales[i]
                score_dedo = bajada_relativa + flow_ventana

                if frames_activos[i] >= 5 and not movimiento_bloque:
                    if not es_acostada:
                        # Mano derecha (vertical)
                        if not dedo_presionado[i]:
                            if flow_ventana > UMBRAL_BAJADA_VERTICAL or bajada_relativa > UMBRAL_BAJADA_VERTICAL:
                                if tocar_sonido_idx(i):
                                    dedo_presionado[i] = True
                        else:
                            if flow_ventana < (UMBRAL_BAJADA_VERTICAL * 0.25):
                                dedo_presionado[i] = False
                    else:
                        # Mano acostada (Standby): solo toca el dedo que baje claramente hacia abajo
                        if not dedo_presionado[i]:
                            if (
                                score_dedo == max_score_mano
                                and score_dedo > UMBRAL_BAJADA_ACOSTADA
                                and flow_ventana > (UMBRAL_BAJADA_ACOSTADA * 0.45)
                            ):
                                if tocar_sonido_idx(i):
                                    dedo_presionado[i] = True
                        else:
                            if score_dedo < (UMBRAL_BAJADA_ACOSTADA * 0.35) or flow_ventana < -0.005:
                                dedo_presionado[i] = False
                                baseline_rel_y[i] = (puntas_xy[i][1] / float(h)) - mediana_y_mano

                # Dibujar SOLO el punto en la punta del dedo y su nota (Do, Re, Mi, Fa, Sol...)
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

        prev_gray = gray

        # Dibujar el pentagrama arriba a la derecha
        dibujar_pentagrama(frame, ahora, alguna_acostada)

        cv2.imshow("Piano Virtual - Solo Puntas de Dedo + Pentagrama", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
        elif key in (ord('+'), ord('=')):
            UMBRAL_BAJADA_VERTICAL = max(0.008, UMBRAL_BAJADA_VERTICAL - 0.002)
            UMBRAL_BAJADA_ACOSTADA = max(0.014, UMBRAL_BAJADA_ACOSTADA - 0.003)
        elif key in (ord('-'), ord('_')):
            UMBRAL_BAJADA_VERTICAL = min(0.040, UMBRAL_BAJADA_VERTICAL + 0.002)
            UMBRAL_BAJADA_ACOSTADA = min(0.065, UMBRAL_BAJADA_ACOSTADA + 0.003)

    cap.release()
    cv2.destroyAllWindows()
    pygame.mixer.quit()

if __name__ == "__main__":
    main()
