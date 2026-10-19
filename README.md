# Piano con las Manos — Postura Real de Piano ("Araña Acostada")

Resumen
-------
Este proyecto implementa un **piano virtual** controlado por las manos usando visión por computador (**OpenCV + MediaPipe Hands**) y **Pygame** para la reproducción de samples de audio (`.mp3` / `.wav`), guiado a partir de [adiacla/piano](https://github.com/adiacla/piano.git) y adaptado específicamente para permitir **apoyar las manos sobre una superficie/mesa en postura de piano real ("araña acostada")** sin que el apoyo afecte la detección ni dispare notas falsas.

¿Cómo resuelve el problema de las manos apoyadas ("Araña Acostada")?
--------------------------------------------------------------------
En un detector básico que solo mide el cambio vertical absoluto (`dy`) de las puntas en un solo frame:
1. Al bajar o apoyar la mano entera sobre la mesa, todas las notas se disparan por accidente.
2. Con la mano acostada horizontalmente (palma hacia abajo y dedos arqueados en forma de araña), el recorrido vertical de cada dedo es corto y las sombras bajo la mano dificultan la detección o invierten la clasificación `Left`/`Right` de MediaPipe.

Esta implementación incorpora **4 mejoras biomecánicas y de visión**:

1. **Compensación del Puente de Soporte (Nudillos `MCP` `[2, 5, 9, 13, 17]`)**:
   - El sistema rastrea el "cuerpo de la araña" (el puente formado por la muñeca y los nudillos base `MCP`).
   - Si el usuario baja toda la mano para apoyarla o acomodarla sobre la mesa, el desplazamiento vertical del puente se resta automáticamente del movimiento de las puntas (`dy_neto`). Así puedes apoyar y descansar las manos sin activar notas por error.
2. **Detección Articular Relativa (`TIP` respecto a `MCP` y `PIP`)**:
   - Cuando la mano está apoyada como araña, pulsar una tecla implica flexionar la punta del dedo (`4, 8, 12, 16, 20`) hacia la superficie respecto a su nudillo (`2, 5, 9, 13, 17`) y articulación media (`3, 6, 10, 14, 18`), incluso cuando el movimiento en píxeles es pequeño por la perspectiva frontal o diagonal de la cámara.
3. **Máquina de Estados con Histéresis (`REPOSO` ↔ `PRESIONADO`) y Ventana Multi-frame**:
   - Evalúa la articulación en una ventana corta de frames y mantiene una línea base adaptativa de reposo (`baseline_rel_y`). Una vez que un dedo toca la mesa y suena la nota, puede quedarse apoyado sin re-disparar el sonido hasta que se levante ligeramente.
4. **Realce de Contraste Local (CLAHE) + Ordenamiento Espacial Inmune a Inversión**:
   - Mejora la iluminación bajo el arco de los dedos apoyados y ordena las manos por su geometría horizontal en pantalla para evitar que MediaPipe confunda la mano izquierda con la derecha cuando las palmas miran hacia abajo.

Mapeo de Notas (10 Dedos)
-------------------------
De izquierda a derecha:
- **Mano Izquierda** (Meñique → Pulgar): `E5`, `D5`, `C5`, `B4`, `A4`
- **Mano Derecha** (Pulgar → Meñique): `C4`, `D4`, `E4`, `F4`, `G4`

Requisitos e Instalación
------------------------
```bash
pip install -r requirements.txt
```

Ejecución
---------
```bash
python piano.py
```

### Controles en tiempo real (en la ventana de video)
- **`+` / `-`**: Aumentar o disminuir la sensibilidad de pulsación según la distancia a la cámara.
- **`M`**: Alternar modo espejo (flip horizontal).
- **`ESC`**: Salir del programa.
