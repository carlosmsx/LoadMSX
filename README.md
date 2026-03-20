# LoadMSX
OCR personalizado para copiar listados BASIC desde revistas Load MSX (y posiblemente otras)

Herramienta en Python para refinar OCR de impresiones matriciales monoespaciadas, especialmente listados BASIC/ASM con datos hexadecimales separados por comas.

La idea no es hacer OCR clásico de una sola vez, sino trabajar con una grilla de caracteres y permitir corrección asistida, guardando ejemplos para reutilizarlos y mejorar progresivamente el reconocimiento.

## Características actuales

- Carga de imagen
- Binarización con umbral ajustable
- Definición manual de grilla
- Selección de celdas con mouse
- Visualización ampliada del carácter
- Ingreso manual del carácter correcto
- Guardado de plantillas aprendidas
- Reconstrucción de texto
- Exportación a archivo `.txt`

## Caso de uso

Pensado para imágenes de:

- listados BASIC
- bloques `DATA`
- rutinas Z80/ASM
- impresiones matriciales
- texto monoespaciado con errores típicos de OCR

Por ejemplo, confusiones como:

- `8` ↔ `B`
- `0` ↔ `C` / `O`
- `,` ↔ `.`
- `'` omitido
- `1` ↔ `I`

## Requisitos

- Python 3.10 o superior
- Pillow

Instalación:

```bash
pip install -r requirements.txt
