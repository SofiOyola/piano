import warnings
# Suprimir advertencia protobuf temprano
warnings.filterwarnings("ignore", message=".*SymbolDatabase.GetPrototype.*")

import cv2
import mediapipe as mp
import numpy as np
import pygame
import os
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
# CONFIGURACIÓN DE NOTAS
# --------------------------
mano_derecha = ['C4', 'D4', 'E4', 'F4', 'G4']
mano_izquierda = ['A4', 'B4', 'C5', 'D5', 'E5']
# Orden de izquierda a derecha (10 dedos):
# meñique izq -> anular izq -> medio izq -> índice izq -> pulgar izq ->
# pulgar der -> índice der -> medio der -> anular der -> meñique der
nota_por_dedo = mano_izquierda[::-1] + mano_derecha  # 10 notas
NOMBRES_DEDOS = [
    "Menique Izq", "Anular Izq", "Medio Izq", "Indice Izq", "Pulgar Izq",
    "Pulgar Der", "Indice Der", "Medio Der", "Anular Der", "Menique Der"
]

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ruta_sonidos = os.path.join(BASE_DIR, "sonidos")

# Frecuencias base por si se requiere respaldo sintetizado
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
    """Genera un tono tipo piano acústico como fallback si un archivo de audio falla."""
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    # Armónicos de piano con envolvente exponencial (ADSR simplificado)
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
for i, nombre in enumerate(nota_por_dedo):
    mp3_path = os.path.join(ruta_sonidos, f"{nombre}.mp3")
    wav_path = os.path.join(ruta_sonidos, f"{nombre}.wav")
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
                    out_wav = os.path.join(TMP_DIR, f"{nombre}.wav")
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
        print(f"ℹ️ Usando tono sintetizado de respaldo para {nombre}")
        sonidos[i] = sintetizar_tono_piano(FRECUENCIAS_NOTAS.get(nombre, 440.0))

# --------------------------
# MEDIAPIPE HANDS (Optimizado para postura "Araña Acostada" / Piano)
# --------------------------
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

# En postura de "araña acostada" (manos apoyadas con dedos arqueados sobre la mesa),
# existe escorzo y sombras bajo los dedos. Usamos umbrales de confianza de 0.45
# combinados con mejora de contraste CLAHE para no perder el tracking al apoyar la mano.
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    model_complexity=1,
    min_detection_confidence=0.45,
    min_tracking_confidence=0.45
)

# Landmarks clave por dedo: (thumb, index, middle, ring, pinky)
FINGER_TIPS = [4, 8, 12, 16, 20]  # Puntas de los dedos
FINGER_PIPS = [3, 6, 10, 14, 18]  # Articulaciones medias (arco de la araña)
FINGER_MCPS = [2, 5, 9, 13, 17]   # Nudillos base (puente de soporte de la mano)

# Preprocesador CLAHE para mejorar la visibilidad de los dedos apoyados en superficie
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

def mejorar_contraste_manos(bgr_frame):
    """Realza el contraste local para que MediaPipe detecte dedos arqueados sobre una mesa."""
    lab = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l_eq = clahe.apply(l)
    lab_eq = cv2.merge((l_eq, a, b))
    return cv2.cvtColor(lab_eq, cv2.COLOR_LAB2RGB)

# --------------------------
# DETECCIÓN DE PULSACIÓN EN POSTURA "ARAÑA ACOSTADA"
# --------------------------
# Cuando el usuario apoya las manos en forma de araña sobre una mesa:
# 1. El puente de nudillos (MCP) sirve de referencia de soporte de la mano.
# 2. Si toda la mano baja para apoyarse en la mesa, todos los MCP bajan al mismo tiempo.
#    Restamos ese desplazamiento global del puente para que APOYAR LA MANO NO DISRARE NOTAS.
# 3. Medimos la articulación propia de cada dedo (caída de la punta respecto a su nudillo
#    y respecto a su posición de reposo en una ventana de frames corta) con histéresis
#    (estado presionado / liberado) para que dejar los dedos apoyados no cause retriggers.

alpha_smooth = 0.65            # Suavizado EMA para filtrar micro-vibraciones de cámara
UMBRAL_MOV_REL = 0.016         # Umbral de descenso neto del dedo (ajustable con +/- en vivo)
UMBRAL_FLEXION_REL = 0.011     # Umbral de flexión articular (TIP respecto a su MCP/PIP)
FACTOR_LIBERACION = 0.45       # Histéresis: fracción del umbral para liberar la tecla
COOLDOWN = 0.16                # Segundos mínimos entre pulsaciones del mismo dedo
VENTANA_FRAMES = 3             # Ventana temporal (frames) para detectar pulsaciones reales

# Estado por cada uno de los 10 dedos
prev_tip_y = [None] * 10
prev_rel_y = [None] * 10       # (y_tip - y_mcp) suavizado
hist_tip_y = [deque(maxlen=VENTANA_FRAMES + 1) for _ in range(10)]
hist_rel_y = [deque(maxlen=VENTANA_FRAMES + 1) for _ in range(10)]
baseline_rel_y = [None] * 10   # Línea base de reposo cuando el dedo descansa apoyado
dedo_presionado = [False] * 10
last_play = {i: 0.0 for i in range(10)}
ultimo_trigger_visual = {i: 0.0 for i in range(10)}

# Historial del puente de cada mano (0: mano izq, 1: mano der) para filtrar apoyo global
prev_bridge_y = [None, None]

def reset_finger_state(idx):
    prev_tip_y[idx] = None
    prev_rel_y[idx] = None
    hist_tip_y[idx].clear()
    hist_rel_y[idx].clear()
    baseline_rel_y[idx] = None
    dedo_presionado[idx] = False

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
# UTIL: Obtener manos y dedos ordenados espacialmente (inmune a inversión en mesa)
# --------------------------
def extraer_datos_mano(lm, frame_width, frame_height, invertir_orden):
    """
    Extrae para los 5 dedos de una mano:
    (x_tip_px, y_tip_px, x_mcp_px, y_mcp_px, x_pip_px, y_pip_px, y_tip_norm, y_mcp_norm, y_pip_norm)
    Además calcula la altura media del puente de nudillos (bridge_y) para compensar el apoyo.
    """
    indices = list(reversed( range(5) )) if invertir_orden else list(range(5))
    # Altura del puente de la mano (mediana de muñeca + nudillos MCP)
    mcp_ys = [lm.landmark[m].y for m in FINGER_MCPS] + [lm.landmark[0].y]
    bridge_y = float(np.median(mcp_ys))

    dedos_info = []
    for k in indices:
        tip_id = FINGER_TIPS[k]
        pip_id = FINGER_PIPS[k]
        mcp_id = FINGER_MCPS[k]

        x_tip = int(lm.landmark[tip_id].x * frame_width)
        y_tip = int(lm.landmark[tip_id].y * frame_height)
        x_pip = int(lm.landmark[pip_id].x * frame_width)
        y_pip = int(lm.landmark[pip_id].y * frame_height)
        x_mcp = int(lm.landmark[mcp_id].x * frame_width)
        y_mcp = int(lm.landmark[mcp_id].y * frame_height)

        y_tip_norm = lm.landmark[tip_id].y
        y_pip_norm = lm.landmark[pip_id].y
        y_mcp_norm = lm.landmark[mcp_id].y

        dedos_info.append({
            "tip_px": (x_tip, y_tip),
            "pip_px": (x_pip, y_pip),
            "mcp_px": (x_mcp, y_mcp),
            "y_tip_norm": y_tip_norm,
            "y_pip_norm": y_pip_norm,
            "y_mcp_norm": y_mcp_norm,
            "bridge_y": bridge_y
        })

    # Si los dedos están apoyados horizontalmente (postura de araña), verificamos que
    # el orden horizontal en pantalla sea consistente de izquierda a derecha:
    xs_tips = [d["tip_px"][0] for d in dedos_info]
    # Si están claramente invertidos en el eje X (ej. palma hacia abajo vs hacia arriba),
    # usamos la anatomía del pulgar vs meñique para asegurar el mapeo correcto.
    return dedos_info, bridge_y

def obtener_tips_ordenados(results, frame_width, frame_height):
    """
    Retorna:
      - lista de 10 diccionarios (uno por dedo en orden meñique izq -> ... -> meñique der),
        o None en las posiciones de la mano que no esté presente.
      - lista de 2 valores bridge_y [bridge_izq, bridge_der] para compensar el apoyo en mesa.

    En postura de "araña acostada" (palmas hacia abajo sobre la mesa), MediaPipe suele
    confundir el label 'Left'/'Right'. Por ello:
      - Si hay 2 manos detectadas, la que está a la izquierda en la imagen es la mano
        izquierda y la de la derecha es la mano derecha.
      - Si hay 1 sola mano, combinamos su posición X respecto al centro y la geometría
        del pulgar (landmark 4) respecto al meñique (landmark 20).
    """
    if not results.multi_hand_landmarks:
        return [None] * 10, [None, None]

    manos_detectadas = []
    for idx_h, lm in enumerate(results.multi_hand_landmarks):
        # Centro X de la mano usando muñeca y nudillos
        cx = float(np.mean([lm.landmark[j].x for j in [0, 5, 9, 13, 17]]))
        label_mp = "Right"
        if results.multi_handedness and idx_h < len(results.multi_handedness):
            label_mp = results.multi_handedness[idx_h].classification[0].label
        manos_detectadas.append((cx, label_mp, lm))

    # Ordenar manos de izquierda a derecha en pantalla
    manos_detectadas.sort(key=lambda item: item[0])

    mano_izq_lm = None
    mano_der_lm = None

    if len(manos_detectadas) >= 2:
        mano_izq_lm = manos_detectadas[0][2]
        mano_der_lm = manos_detectadas[1][2]
    elif len(manos_detectadas) == 1:
        cx, label_mp, lm = manos_detectadas[0]
        # En postura de araña (palma abajo, vista en espejo):
        # Mano izquierda tiene el pulgar (4) a la derecha del meñique (20): lm[4].x > lm[20].x
        # Mano derecha tiene el pulgar (4) a la izquierda del meñique (20): lm[4].x < lm[20].x
        thumb_x = lm.landmark[4].x
        pinky_x = lm.landmark[20].x
        if abs(thumb_x - pinky_x) > 0.04:
            if thumb_x > pinky_x:
                mano_izq_lm = lm
            else:
                mano_der_lm = lm
        else:
            if cx < 0.5:
                mano_izq_lm = lm
            else:
                mano_der_lm = lm

    salida = []
    bridges = [None, None]

    # Mano izquierda: queremos meñique (20) -> ... -> pulgar (4) (invertir_orden=True)
    if mano_izq_lm is not None:
        dedos_izq, b_izq = extraer_datos_mano(mano_izq_lm, frame_width, frame_height, invertir_orden=True)
        salida.extend(dedos_izq)
        bridges[0] = b_izq
    else:
        salida.extend([None] * 5)

    # Mano derecha: queremos pulgar (4) -> ... -> meñique (20) (invertir_orden=False)
    if mano_der_lm is not None:
        dedos_der, b_der = extraer_datos_mano(mano_der_lm, frame_width, frame_height, invertir_orden=False)
        salida.extend(dedos_der)
        bridges[1] = b_der
    else:
        salida.extend([None] * 5)

    return salida[:10], bridges

def dibujar_arana_mano(frame, tips_data, start_idx, end_idx, ahora):
    """
    Dibuja la estructura en 'araña acostada' (puente de nudillos MCP + arcos PIP->TIP)
    para visualizar el soporte de la mano sin que afecte la detección.
    """
    mcps_validos = []
    for i in range(start_idx, end_idx):
        d = tips_data[i]
        if d is None:
            continue
        mcp_px = d["mcp_px"]
        pip_px = d["pip_px"]
        tip_px = d["tip_px"]
        mcps_validos.append(mcp_px)

        # Color según si el dedo acaba de tocar o está apoyado en reposo
        reciente = (ahora - ultimo_trigger_visual.get(i, 0.0)) < 0.22
        color_arco = (0, 255, 0) if (dedo_presionado[i] or reciente) else (255, 200, 0)
        color_punta = (0, 255, 0) if (dedo_presionado[i] or reciente) else (255, 80, 0)

        # Dibujar falanges arqueadas (patas de la araña: MCP -> PIP -> TIP)
        cv2.line(frame, mcp_px, pip_px, (180, 180, 180), 2)
        cv2.line(frame, pip_px, tip_px, color_arco, 2)
        cv2.circle(frame, mcp_px, 4, (200, 200, 200), -1)
        cv2.circle(frame, pip_px, 4, (0, 215, 255), -1)
        cv2.circle(frame, tip_px, 7, color_punta, -1)

        # Etiqueta pequeña de la nota asignada a ese dedo
        nota = nota_por_dedo[i]
        cv2.putText(
            frame, nota,
            (tip_px[0] - 12, tip_px[1] + 22),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45,
            (0, 255, 0) if reciente else (230, 230, 230), 1, cv2.LINE_AA
        )

        if reciente:
            cv2.putText(
                frame, f"♪ {nota}",
                (tip_px[0] - 18, tip_px[1] - 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (0, 255, 0), 2, cv2.LINE_AA
            )

    # Conectar el puente de los nudillos (cuerpo de la araña)
    for k in range(len(mcps_validos) - 1):
        cv2.line(frame, mcps_validos[k], mcps_validos[k + 1], (160, 160, 160), 2)

# --------------------------
# BUCLE PRINCIPAL
# --------------------------
def main():
    global UMBRAL_MOV_REL, UMBRAL_FLEXION_REL

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("No se puede abrir la cámara.")
        return

    print("\n🎹 Piano Virtual Activado — Modo 'Araña Acostada' (Soporte en Superficie)")
    print("-------------------------------------------------------------------------")
    print("• Puedes apoyar las manos arqueadas sobre la mesa como en un piano real.")
    print("• El sistema compensa el apoyo de la mano para no disparar notas al descansar.")
    print("• Controles en ventana:")
    print("   [+] / [-] : Aumentar / disminuir sensibilidad de pulsación")
    print("   [M]       : Alternar vista espejo (flip horizontal)")
    print("   [ESC]     : Salir\n")

    usar_espejo = True

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error leyendo frame.")
            break

        if usar_espejo:
            frame = cv2.flip(frame, 1)

        h, w, _ = frame.shape

        # Mejorar contraste local para detectar manos apoyadas / arqueadas sin perder dedos
        rgb_mejorado = mejorar_contraste_manos(frame)
        results = hands.process(rgb_mejorado)

        ahora = time.time()

        # Si no hay manos detectadas, limpiar historial
        if not results.multi_hand_landmarks:
            for i in range(10):
                reset_finger_state(i)
            prev_bridge_y[0] = None
            prev_bridge_y[1] = None

            cv2.putText(
                frame, "Coloca tus manos en postura de piano (arana acostada)",
                (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 200, 255), 2, cv2.LINE_AA
            )
            cv2.imshow("Piano - Postura Arana Acostada (pygame + MediaPipe)", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break
            elif key in (ord('m'), ord('M')):
                usar_espejo = not usar_espejo
            continue

        tips_data, bridges = obtener_tips_ordenados(results, w, h)

        # Calcular el movimiento vertical global de cada mano (puente de nudillos)
        # Si toda la mano está bajando para apoyarse sobre la mesa, compensamos ese
        # movimiento para que apoyar la mano no active las notas por error.
        dy_bridge_mano = [0.0, 0.0]
        for m_idx in (0, 1):
            if bridges[m_idx] is not None:
                if prev_bridge_y[m_idx] is not None:
                    dy_bridge_mano[m_idx] = bridges[m_idx] - prev_bridge_y[m_idx]
                prev_bridge_y[m_idx] = bridges[m_idx]
            else:
                prev_bridge_y[m_idx] = None

        # Evaluar las 10 puntas con compensación de apoyo + flexión articular relativa
        deltas_mano = {0: [], 1: []}
        candidatos_disparo = []

        for i in range(10):
            mano_idx = 0 if i < 5 else 1
            d = tips_data[i]
            if d is None:
                reset_finger_state(i)
                continue

            y_tip_norm = d["y_tip_norm"]
            y_mcp_norm = d["y_mcp_norm"]
            y_pip_norm = d["y_pip_norm"]

            # Distancia vertical relativa de la punta respecto a su arco (nudillo MCP y PIP)
            # En postura de araña acostada, cuando un dedo presiona una tecla, la punta baja
            # acercándose o superando el nivel inferior respecto a su nudillo/arco.
            rel_y_raw = (0.65 * (y_tip_norm - y_mcp_norm)) + (0.35 * (y_tip_norm - y_pip_norm))

            if prev_tip_y[i] is None:
                y_smooth = y_tip_norm
                rel_smooth = rel_y_raw
                baseline_rel_y[i] = rel_smooth
            else:
                y_smooth = alpha_smooth * y_tip_norm + (1.0 - alpha_smooth) * prev_tip_y[i]
                rel_smooth = alpha_smooth * rel_y_raw + (1.0 - alpha_smooth) * prev_rel_y[i]

            hist_tip_y[i].append(y_smooth)
            hist_rel_y[i].append(rel_smooth)

            if len(hist_tip_y[i]) >= 2:
                # Movimiento en ventana corta de frames (permite detectar pulsaciones
                # reales de piano tanto rápidas como suaves con la mano apoyada)
                dy_window = y_smooth - hist_tip_y[i][0]
                drel_window = rel_smooth - hist_rel_y[i][0]

                # Compensar bajada global de la mano al apoyarse en la mesa
                bajada_global_mano = max(0.0, dy_bridge_mano[mano_idx]) * len(hist_tip_y[i]) * 0.75
                dy_neto = dy_window - bajada_global_mano

                # Actualizar suavemente la línea base de reposo cuando el dedo está quieto apoyado
                if abs(drel_window) < (UMBRAL_FLEXION_REL * 0.4) and not dedo_presionado[i]:
                    baseline_rel_y[i] = 0.92 * baseline_rel_y[i] + 0.08 * rel_smooth

                desplazamiento_desde_reposo = rel_smooth - baseline_rel_y[i]
                deltas_mano[mano_idx].append((i, dy_neto, drel_window, desplazamiento_desde_reposo))

            prev_tip_y[i] = y_smooth
            prev_rel_y[i] = rel_smooth

        # Decidir pulsaciones filtrando si toda la mano se movió en bloque al apoyarse
        for mano_idx in (0, 1):
            lista_d = deltas_mano[mano_idx]
            if not lista_d:
                continue

            # Si los 5 dedos bajan casi idéntico al mismo tiempo sin flexión diferencial,
            # es la mano acomodándose sobre la mesa -> mediana de flexión de la mano
            mediana_drel = float(np.median([item[2] for item in lista_d])) if len(lista_d) >= 3 else 0.0

            for (i, dy_neto, drel_window, desp_reposo) in lista_d:
                # Flexión diferencial del dedo respecto al resto de la mano apoyada
                flexion_diferencial = drel_window - (0.5 * mediana_drel if mediana_drel > 0 else 0.0)

                if not dedo_presionado[i]:
                    # Condición de pulsación en postura "araña acostada":
                    # O bien hay un descenso neto claro acompañado de flexión articular,
                    # o bien una flexión articular marcada respecto al puente de la mano apoyada.
                    cond_velocidad = (dy_neto > UMBRAL_MOV_REL) and (flexion_diferencial > UMBRAL_FLEXION_REL * 0.65)
                    cond_articulacion = (flexion_diferencial > UMBRAL_FLEXION_REL) and (desp_reposo > UMBRAL_FLEXION_REL * 0.85) and (dy_neto > UMBRAL_MOV_REL * 0.45)

                    if cond_velocidad or cond_articulacion:
                        if tocar_sonido_idx(i):
                            dedo_presionado[i] = True
                            baseline_rel_y[i] = prev_rel_y[i] - (UMBRAL_FLEXION_REL * 0.5)
                else:
                    # Histéresis para liberar el dedo: cuando el dedo deja de bajar o sube levemente
                    # permite mantener la mano apoyada sin re-disparar la nota.
                    if drel_window < (UMBRAL_FLEXION_REL * FACTOR_LIBERACION) and dy_neto < (UMBRAL_MOV_REL * FACTOR_LIBERACION):
                        dedo_presionado[i] = False

        # Dibujar visualización de ambas manos en postura de araña
        dibujar_arana_mano(frame, tips_data, 0, 5, ahora)
        dibujar_arana_mano(frame, tips_data, 5, 10, ahora)

        # HUD informativo
        cv2.rectangle(frame, (0, 0), (w, 36), (25, 25, 25), -1)
        hud_txt = f"Modo Arana Acostada (Mesa) | Sensibilidad (+/-): {UMBRAL_MOV_REL:.3f} | Espejo (M): {'SI' if usar_espejo else 'NO'}"
        cv2.putText(frame, hud_txt, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (220, 255, 220), 1, cv2.LINE_AA)

        cv2.imshow("Piano - Postura Arana Acostada (pygame + MediaPipe)", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
        elif key in (ord('+'), ord('=')):
            # Más sensible (umbral más bajo)
            UMBRAL_MOV_REL = max(0.006, UMBRAL_MOV_REL - 0.002)
            UMBRAL_FLEXION_REL = max(0.005, UMBRAL_FLEXION_REL - 0.0015)
        elif key in (ord('-'), ord('_')):
            # Menos sensible (umbral más alto)
            UMBRAL_MOV_REL = min(0.045, UMBRAL_MOV_REL + 0.002)
            UMBRAL_FLEXION_REL = min(0.035, UMBRAL_FLEXION_REL + 0.0015)
        elif key in (ord('m'), ord('M')):
            usar_espejo = not usar_espejo

    cap.release()
    cv2.destroyAllWindows()
    pygame.mixer.quit()

if __name__ == "__main__":
    main()
