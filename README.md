# Piano con las Manos — Mano Horizontal y Pentagrama en Vivo

Resumen
-------
Este proyecto implementa un **piano virtual** controlado con las manos usando visión por computador (**OpenCV + MediaPipe Hands**) y **Pygame** para reproducir samples de piano (`.mp3` / `.wav`).

Características principales
---------------------------
1. **Detección en postura de mano horizontal ("araña acostada")**:
   - Ordena automáticamente las 5 puntas de cada mano de **izquierda a derecha en el eje X**, evitando que las líneas o los dedos se crucen cuando colocas la mano horizontalmente con la palma hacia abajo.
   - Al **bajar un dedo**, suena inmediatamente su nota asignada. Si mueves o apoyas toda la mano en bloque, se compensa el movimiento común para evitar que suenen todas las notas por accidente.
2. **Notación de Piano Tradicional (`Do`, `Re`, `Mi`, `Fa`, `Sol`, `La`, `Si`)**:
   - En lugar de cifrado inglés de guitarra (`C, D, E...`), muestra los nombres clásicos de piano en español directamente debajo de cada dedo:
     - **Mano 1** (de izquierda a derecha): `Do`, `Re`, `Mi`, `Fa`, `Sol`
     - **Mano 2** (de izquierda a derecha): `La`, `Si`, `Do`, `Re`, `Mi`
3. **Pentagrama en Tiempo Real (Esquina Superior Derecha)**:
   - Muestra un pentagrama musical con sus 5 líneas en Clave de Sol en la esquina superior derecha de la pantalla.
   - Cada vez que tocas una nota, se ilumina un **círculo (cabeza de nota musical)** exactamente en la línea o espacio correspondiente del pentagrama (incluyendo la línea adicional inferior para el `Do` central).

Instalación y Ejecución
-----------------------
```bash
pip install -r requirements.txt
python piano.py
```

### Controles en ventana
- **`+` / `-`**: Aumentar o disminuir la sensibilidad al bajar el dedo.
- **`ESC`**: Salir del programa.
