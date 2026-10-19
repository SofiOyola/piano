# Piano Virtual con las Manos

<div align="center">

### Piano interactivo controlado mediante visión por computador

**Python | OpenCV | MediaPipe Hands | Pygame | NumPy**

Controla notas musicales con el movimiento de tus dedos frente a una cámara y observa su representación en un pentagrama en tiempo real.

</div>

---

## Descripción

**Piano Virtual con las Manos** es una aplicación interactiva desarrollada en Python que permite tocar notas musicales utilizando los movimientos de los dedos frente a una cámara.

El sistema utiliza visión por computador para detectar hasta dos manos, identificar la posición de los dedos y reconocer cuándo uno de ellos realiza un movimiento de pulsación. Al detectar este movimiento, reproduce la nota musical correspondiente.

Además de generar el sonido, la aplicación muestra las notas sobre la imagen de la cámara e incorpora un pentagrama en tiempo real que permite visualizar qué nota está siendo interpretada.

---

## Objetivo del proyecto

El objetivo principal del proyecto es desarrollar una forma de interacción musical sin necesidad de utilizar un teclado físico, combinando visión por computador, seguimiento de manos y reproducción de audio.

La aplicación busca transformar los movimientos naturales de los dedos en entradas musicales, permitiendo:

- Detectar las manos mediante una cámara.
- Identificar individualmente los dedos.
- Asociar cada dedo con una nota musical.
- Reconocer el movimiento de pulsación de un dedo.
- Reproducir sonidos de piano en tiempo real.
- Mostrar visualmente las notas asignadas.
- Representar las notas interpretadas en un pentagrama.
- Ajustar la sensibilidad del reconocimiento durante la ejecución.

---

## Funcionamiento general

El flujo principal de la aplicación es:

```text
Cámara
   |
   v
OpenCV
   |
   v
MediaPipe Hands
   |
   v
Detección de hasta 2 manos
   |
   v
Seguimiento de los dedos
   |
   v
Detección del movimiento de pulsación
   |
   +-------------------+
   |                   |
   v                   v
Reproducción       Pentagrama
de sonido          en tiempo real
   |
   v
Pygame
```

OpenCV captura continuamente la imagen de la cámara. Posteriormente, MediaPipe analiza cada fotograma para encontrar las manos y sus puntos de referencia.

El programa ordena las puntas de los dedos horizontalmente y analiza su movimiento vertical. Cuando detecta que un dedo baja de forma independiente respecto al movimiento general de la mano, interpreta la acción como una pulsación de piano.

---

## Notas disponibles

El piano utiliza diez posiciones correspondientes a los diez dedos de dos manos.

| Mano | Posición de los dedos | Notas |
|---|---|---|
| Mano 1 | De izquierda a derecha | Do - Re - Mi - Fa - Sol |
| Mano 2 | De izquierda a derecha | La - Si - Do - Re - Mi |

Internamente, las notas corresponden a:

```text
C4 - D4 - E4 - F4 - G4 - A4 - B4 - C5 - D5 - E5
```

La aplicación muestra los nombres tradicionales en español:

```text
Do - Re - Mi - Fa - Sol - La - Si - Do - Re - Mi
```

---

## Requisitos

Antes de ejecutar el proyecto es necesario disponer de:

| Requisito | Descripción |
|---|---|
| Python | Python 3.10, 3.11 o 3.12 |
| Versión recomendada | Python 3.11 |
| Cámara | Webcam integrada o externa |
| pip | Administrador de paquetes de Python |
| Audio | Dispositivo de salida de sonido |
| Git | Opcional, recomendado para clonar el proyecto |

### Versión de Python

El proyecto actualmente no define una versión exacta de Python mediante archivos como:

```text
.python-version
pyproject.toml
runtime.txt
```

Tampoco fija versiones específicas dentro de `requirements.txt`.

Debido principalmente a la compatibilidad de **MediaPipe**, se recomienda trabajar con:

```text
Python 3.10 - 3.12
```

La versión recomendada para ejecutar el proyecto es:

```text
Python 3.11
```

No se recomienda utilizar versiones más recientes sin comprobar primero que MediaPipe y las demás dependencias funcionen correctamente.

---

## Librerías utilizadas

El archivo `requirements.txt` contiene cuatro dependencias principales:

```text
opencv-python
mediapipe
pygame
numpy
```

### OpenCV

```text
opencv-python
```

Se utiliza para:

- Acceder a la cámara.
- Capturar los fotogramas.
- Mostrar la interfaz.
- Dibujar textos, líneas, botones y elementos gráficos.
- Mostrar la imagen en modo espejo.
- Gestionar eventos de teclado y ratón.

---

### MediaPipe

```text
mediapipe
```

Se utiliza para detectar y realizar seguimiento de las manos.

MediaPipe proporciona los puntos de referencia necesarios para conocer la posición de los dedos y determinar cuándo se realiza una pulsación.

El proyecto está configurado para detectar hasta:

```text
2 manos
```

simultáneamente.

---

### Pygame

```text
pygame
```

Se utiliza principalmente para el sistema de audio.

Permite:

- Cargar sonidos.
- Reproducir las notas del piano.
- Utilizar varios canales de audio.
- Reproducir varias notas con poca latencia.

---

### NumPy

```text
numpy
```

Se utiliza para operaciones numéricas y procesamiento de señales.

También permite generar sonidos de respaldo cuando un archivo de audio no puede cargarse correctamente.

---

## Dependencias opcionales de audio

El código también puede utilizar:

```text
pydub
ffmpeg
```

Estas herramientas **no son obligatorias para ejecutar el proyecto** y actualmente no aparecen en `requirements.txt`.

Su función es ayudar a convertir archivos MP3 a WAV cuando Pygame no puede cargar directamente algún archivo MP3.

El programa cuenta además con un mecanismo de respaldo capaz de sintetizar una nota mediante NumPy si no puede cargar el archivo de sonido correspondiente.

---

## Instalación

### 1. Clonar el repositorio

Desde una terminal:

```bash
git clone <URL_DEL_REPOSITORIO>
```

Luego ingresa a la carpeta:

```bash
cd piano
```

---

### 2. Crear un entorno virtual

Es recomendable utilizar un entorno virtual para evitar conflictos con otras instalaciones de Python.

```bash
python -m venv .venv
```

### Windows

```bash
.venv\Scripts\activate
```

### Linux o macOS

```bash
source .venv/bin/activate
```

Cuando esté correctamente activado, normalmente aparecerá:

```text
(.venv)
```

al comienzo de la terminal.

---

### 3. Actualizar pip

```bash
python -m pip install --upgrade pip
```

---

### 4. Instalar las dependencias

Ejecuta:

```bash
pip install -r requirements.txt
```

Esto instalará:

```text
OpenCV
MediaPipe
Pygame
NumPy
```

---

## Cómo ejecutar el proyecto

Con el entorno virtual activo y las dependencias instaladas, ejecuta:

```bash
python piano.py
```

La aplicación intentará acceder automáticamente a la cámara principal del computador.

Se abrirá una ventana llamada:

```text
Piano Virtual - Mano Horizontal + Pentagrama
```

---

## Instrucciones de uso

### 1. Iniciar el piano

Al ejecutar el programa aparecerá un menú inicial.

Haz clic sobre el piano que contiene el botón:

```text
START
```

También puedes iniciar utilizando:

```text
ENTER
```

o:

```text
ESPACIO
```

---

### 2. Colocar las manos

Coloca una o dos manos frente a la cámara.

Para conseguir un mejor reconocimiento, mantén las manos aproximadamente en posición horizontal, con los dedos separados.

La aplicación ordenará los dedos de izquierda a derecha para asignar correctamente las notas.

---

### 3. Tocar una nota

Baja uno de los dedos mientras mantienes relativamente estable el resto de la mano.

El programa compara el movimiento del dedo con el movimiento general de la mano.

Cuando identifica que un dedo bajó de manera independiente:

```text
Movimiento del dedo
        |
        v
Detección de pulsación
        |
        v
Reproducción de la nota
        |
        v
Visualización en pentagrama
```

De esta manera se reduce la posibilidad de que todas las notas se activen accidentalmente cuando se mueve toda la mano.

---

## Pentagrama en tiempo real

En la parte superior derecha de la ventana se muestra un pentagrama.

Cuando se toca una nota:

1. Se reproduce su sonido.
2. Se identifica su nombre.
3. Se representa temporalmente sobre el pentagrama.
4. Se muestra la nota que está siendo interpretada.

El sistema incluye las cinco líneas del pentagrama y una representación de la clave de Sol.

---

## Controles

| Control | Función |
|---|---|
| `START` | Iniciar el piano |
| `ENTER` | Iniciar desde el menú |
| `ESPACIO` | Iniciar desde el menú |
| `+` o `=` | Aumentar la sensibilidad |
| `-` o `_` | Disminuir la sensibilidad |
| `ESC` | Cerrar el programa |
| Botón `Salir` | Cerrar el programa |

---

## Ajuste de sensibilidad

El reconocimiento de una pulsación depende de cuánto debe desplazarse verticalmente un dedo para considerarse una nota.

La aplicación permite modificar este valor durante la ejecución.

### Mayor sensibilidad

Presiona:

```text
+
```

o:

```text
=
```

Esto permite detectar movimientos más pequeños de los dedos.

### Menor sensibilidad

Presiona:

```text
-
```

o:

```text
_
```

Esto exige un movimiento mayor para activar una nota.

Este ajuste puede ser útil dependiendo de:

- La distancia respecto a la cámara.
- La iluminación.
- La posición de las manos.
- La resolución de la cámara.
- La manera particular en que el usuario mueve los dedos.

---

## Estructura del proyecto

```text
piano/
│
├── sonidos/
│   └── Archivos de audio correspondientes a las notas
│
├── .gitignore
│
├── README.md
│
├── piano.py
│
└── requirements.txt
```

### `piano.py`

Es el archivo principal de la aplicación.

Contiene la lógica relacionada con:

- Inicialización de la cámara.
- Procesamiento de imagen.
- MediaPipe Hands.
- Seguimiento de dedos.
- Detección de pulsaciones.
- Reproducción de sonido.
- Generación de tonos de respaldo.
- Pentagrama.
- Menú de inicio.
- Botón de salida.
- Controles de sensibilidad.

---

### `sonidos/`

Contiene los samples utilizados para reproducir las notas del piano.

El programa busca archivos asociados a:

```text
C4
D4
E4
F4
G4
A4
B4
C5
D5
E5
```

Puede trabajar con archivos:

```text
.mp3
.wav
```

Si un sonido no puede cargarse, el programa dispone de un mecanismo para generar una aproximación de la nota de manera automática.

---

### `requirements.txt`

Contiene las dependencias externas necesarias para ejecutar el proyecto:

```text
opencv-python
mediapipe
pygame
numpy
```

---

### `.gitignore`

Indica los archivos y carpetas que Git no debe incluir dentro del repositorio.

---

## Configuración de la cámara

La aplicación utiliza por defecto:

```python
cv2.VideoCapture(0)
```

El índice:

```text
0
```

representa normalmente la cámara principal del computador.

Si el equipo posee varias cámaras y se abre una diferente a la deseada, puede ser necesario modificar `piano.py`.

Por ejemplo:

```python
cv2.VideoCapture(1)
```

o:

```python
cv2.VideoCapture(2)
```

dependiendo del dispositivo.

---

## Recomendaciones de uso

Para mejorar el reconocimiento de los dedos se recomienda:

- Utilizar un espacio con buena iluminación.
- Evitar que la mano quede fuera del campo de visión.
- Separar suficientemente los dedos.
- Mantener las manos en posición horizontal.
- Evitar fondos con demasiado movimiento.
- No realizar movimientos excesivamente rápidos.
- Colocarse a una distancia cómoda de la cámara.
- Ajustar la sensibilidad con `+` y `-` cuando sea necesario.
- Evitar que otras personas coloquen las manos frente a la cámara durante la ejecución.

---

## Posibles problemas

### La cámara no se abre

Si aparece:

```text
No se puede abrir la cámara.
```

comprueba que:

1. La cámara esté conectada.
2. El sistema operativo permita a Python acceder a la cámara.
3. Ninguna otra aplicación esté utilizando el dispositivo.
4. El índice configurado en `VideoCapture()` corresponda a la cámara correcta.

---

### Aparece `ModuleNotFoundError`

Por ejemplo:

```text
ModuleNotFoundError: No module named 'mediapipe'
```

Activa primero el entorno virtual y ejecuta nuevamente:

```bash
pip install -r requirements.txt
```

---

### MediaPipe no se instala

Comprueba la versión de Python:

```bash
python --version
```

Se recomienda:

```text
Python 3.11
```

y trabajar dentro del rango:

```text
Python 3.10 - 3.12
```

---

### No se escuchan las notas

Comprueba:

- El volumen del sistema.
- El dispositivo de salida de audio.
- Que Pygame haya podido inicializar el mezclador.
- Que la carpeta `sonidos/` esté presente.
- Que los archivos de sonido no estén dañados.

El programa puede generar sonidos de respaldo si alguno de los samples no está disponible.

---

### Se activan notas accidentalmente

Prueba disminuyendo la sensibilidad mediante:

```text
-
```

También puedes:

- Mantener más estable la mano.
- Mejorar la iluminación.
- Alejarte o acercarte ligeramente a la cámara.
- Mantener los dedos separados.

---

### Cuesta activar las notas

Aumenta la sensibilidad utilizando:

```text
+
```

De esta forma será necesario un desplazamiento vertical menor para detectar una pulsación.

---

## Limitaciones actuales

El proyecto presenta algunas consideraciones importantes:

- Depende de una cámara para funcionar.
- El reconocimiento puede variar según la iluminación y la posición de las manos.
- Está diseñado principalmente para manos colocadas horizontalmente.
- El sistema detecta un máximo de dos manos.
- Actualmente dispone de diez notas.
- `requirements.txt` no fija versiones concretas de las dependencias.
- La compatibilidad con versiones nuevas de Python depende principalmente de MediaPipe.
- La precisión puede variar dependiendo de la calidad y resolución de la cámara.

Para mantener un entorno estable en futuras versiones se recomienda fijar las versiones de las dependencias una vez se determine una configuración completamente validada.

---

## Resumen de ejecución

Una vez descargado el proyecto:

```bash
cd piano
```

Crear el entorno:

```bash
python -m venv .venv
```

Activarlo en Windows:

```bash
.venv\Scripts\activate
```

Instalar dependencias:

```bash
pip install -r requirements.txt
```

Ejecutar:

```bash
python piano.py
```

Después:

```text
1. Haz clic en START
2. Coloca las manos frente a la cámara
3. Mantén los dedos separados y en posición horizontal
4. Baja un dedo para reproducir su nota
5. Observa la nota en el pentagrama
6. Ajusta la sensibilidad con + o -
7. Presiona ESC o Salir para finalizar
```

---

<div align="center">

## Piano Virtual con las Manos

**Visión por computador, interacción natural y música en una sola aplicación.**

</div>
