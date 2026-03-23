
import json
import os
import sys
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

from PIL import Image, ImageOps, ImageTk, ImageDraw, ImageFont


TEMPLATES_FILE = "templates.json"
PROJECT_EXT = ".ogrid.json"
PREDICTION_MAX_DIST = 70
SPACE_AUTO_DENSITY = 0.035
SPACE_FORCE_DENSITY = 0.018
MAX_SAMPLES_PER_CHAR = 200


def load_overlay_font(size: int) -> ImageFont.ImageFont:
    candidates = [
        "DejaVuSansMono.ttf",
        "DejaVuSans.ttf",
        "arial.ttf",
        "cour.ttf",
    ]
    for name in candidates:
        try:
            return ImageFont.truetype(name, size=size)
        except Exception:
            pass
    return ImageFont.load_default()


@dataclass
class GridParams:
    x0: float = 10.0
    y0: float = 10.0
    cell_w: float = 12.0
    cell_h: float = 16.0
    pitch_x: float = 12.0
    pitch_y: float = 16.0
    drift_x: float = 0.0
    drift_y: float = 0.0
    cols: int = 80
    rows: int = 25
    threshold: int = 160
    zoom: float = 2.0


class TemplateStore:
    def __init__(self, path: str):
        self.path = path
        self.templates: Dict[str, List[Dict]] = {}
        self.load()

    def _normalize_templates(self, raw) -> Dict[str, List[Dict]]:
        normalized: Dict[str, List[Dict]] = {}
        if not isinstance(raw, dict):
            return normalized

        for char, items in raw.items():
            if not isinstance(char, str):
                continue
            out: List[Dict] = []

            if isinstance(items, list):
                for item in items:
                    sig = None
                    if isinstance(item, dict):
                        sig = item.get("sig")
                    elif isinstance(item, list):
                        sig = item

                    if isinstance(sig, list) and all(isinstance(v, (int, float)) for v in sig):
                        sig2 = [int(v) for v in sig]
                        out.append({
                            "sig": sig2,
                            "density": sum(sig2) / max(1, len(sig2))
                        })

            if out:
                normalized[char] = out

        return normalized

    def load(self) -> None:
        if os.path.exists(self.path):
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    raw = json.load(f)
                self.templates = self._normalize_templates(raw)
            except Exception:
                self.templates = {}
        else:
            self.templates = {}

    def save(self) -> None:
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self.templates, f, ensure_ascii=False, indent=2)

    def add_sample(self, char: str, signature: List[int]) -> None:
        items = self.templates.setdefault(char, [])
        sig2 = [int(v) for v in signature]
        density = sum(sig2) / max(1, len(sig2))

        for item in items:
            old = item.get("sig")
            if isinstance(old, list) and len(old) == len(sig2):
                dist = sum(abs(int(a) - int(b)) for a, b in zip(old, sig2))
                if dist <= 2:
                    return

        items.append({"sig": sig2, "density": density})

        if len(items) > MAX_SAMPLES_PER_CHAR:
            self.templates[char] = items[-MAX_SAMPLES_PER_CHAR:]

        self.save()

    def count(self, char: str) -> int:
        items = self.templates.get(char, [])
        return len(items) if isinstance(items, list) else 0

    def predict(self, signature: List[int]) -> Tuple[Optional[str], Optional[float]]:
        target_density = sum(signature) / max(1, len(signature))

        if target_density <= SPACE_FORCE_DENSITY:
            return " ", 0.0

        best_char = None
        best_dist = None

        for char, items in self.templates.items():
            if not isinstance(items, list):
                continue
            for item in items:
                sig = None
                density = None
                if isinstance(item, dict):
                    sig = item.get("sig")
                    density = item.get("density")
                elif isinstance(item, list):
                    sig = item
                if not isinstance(sig, list) or len(sig) != len(signature):
                    continue

                dist = sum(abs(int(a) - int(b)) for a, b in zip(sig, signature))

                if density is None:
                    density = sum(sig) / max(1, len(sig))
                dens_penalty = abs(float(density) - float(target_density)) * 90.0
                dist2 = dist + dens_penalty

                if target_density <= SPACE_AUTO_DENSITY and char != " ":
                    dist2 += 35.0

                if best_dist is None or dist2 < best_dist:
                    best_dist = dist2
                    best_char = char

        if target_density <= SPACE_AUTO_DENSITY and best_char not in (" ", None):
            return " ", (best_dist or 0) + 5.0

        return best_char, best_dist


class OCRRefineApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("OCR Refine Grid")
        self.root.geometry("1560x950")

        self.params = GridParams()
        self.image_path: Optional[str] = None
        self.original_image: Optional[Image.Image] = None
        self.binary_image: Optional[Image.Image] = None
        self.display_image: Optional[Image.Image] = None
        self.display_photo: Optional[ImageTk.PhotoImage] = None
        self.cell_photo: Optional[ImageTk.PhotoImage] = None
        self.prediction_photo: Optional[ImageTk.PhotoImage] = None
        self.context_photo: Optional[ImageTk.PhotoImage] = None
        self.overlay_font = load_overlay_font(16)

        self.selected_cell: Tuple[int, int] = (0, 0)
        self.recognized: Dict[str, str] = {}
        self.prediction_cache: Dict[str, Tuple[Optional[str], Optional[float]]] = {}
        self.template_store = TemplateStore(TEMPLATES_FILE)

        self.reference_lines: List[str] = []
        self.validation_map: Dict[str, bool] = {}

        self.status_var = tk.StringVar(value="Abrí una imagen para empezar.")
        self.prediction_var = tk.StringVar(value="Predicción: -")
        self.image_size_var = tk.StringVar(value="Imagen: -")
        self.show_predictions_var = tk.BooleanVar(value=True)

        self.vars = {
            "x0": tk.DoubleVar(value=self.params.x0),
            "y0": tk.DoubleVar(value=self.params.y0),
            "cell_w": tk.DoubleVar(value=self.params.cell_w),
            "cell_h": tk.DoubleVar(value=self.params.cell_h),
            "pitch_x": tk.DoubleVar(value=self.params.pitch_x),
            "pitch_y": tk.DoubleVar(value=self.params.pitch_y),
            "drift_x": tk.DoubleVar(value=self.params.drift_x),
            "drift_y": tk.DoubleVar(value=self.params.drift_y),
            "cols": tk.IntVar(value=self.params.cols),
            "rows": tk.IntVar(value=self.params.rows),
            "threshold": tk.IntVar(value=self.params.threshold),
            "zoom": tk.DoubleVar(value=self.params.zoom),
        }

        self._build_ui()
        self._bind_keys()
        self.update_template_status()

    def _build_ui(self) -> None:
        top = ttk.Frame(self.root)
        top.pack(fill="both", expand=True)

        controls = ttk.LabelFrame(top, text="Parámetros")
        controls.pack(side="top", fill="x", padx=8, pady=6)

        fields = [
            ("X0", "x0", 0, 0),
            ("Y0", "y0", 0, 3),
            ("Ancho celda", "cell_w", 0, 6),
            ("Alto celda", "cell_h", 0, 9),
            ("Pitch X", "pitch_x", 1, 0),
            ("Pitch Y", "pitch_y", 1, 3),
            ("Drift X/fila", "drift_x", 1, 6),
            ("Drift Y/col", "drift_y", 1, 9),
            ("Columnas", "cols", 2, 0),
            ("Filas", "rows", 2, 3),
            ("Threshold", "threshold", 2, 6),
        ]

        float_keys = {"x0", "y0", "cell_w", "cell_h", "pitch_x", "pitch_y", "drift_x", "drift_y"}
        for label, key, r, c in fields:
            ttk.Label(controls, text=label).grid(row=r, column=c, padx=4, pady=4, sticky="e")
            if key in float_keys:
                widget = ttk.Spinbox(controls, textvariable=self.vars[key], width=8, increment=0.1, from_=-9999, to=9999)
            else:
                widget = ttk.Spinbox(controls, textvariable=self.vars[key], width=8, increment=1, from_=-9999, to=9999)
            widget.grid(row=r, column=c + 1, padx=4, pady=4)

        nudge = ttk.LabelFrame(controls, text="Mover grilla")
        nudge.grid(row=0, column=12, rowspan=3, padx=(12, 4), pady=2, sticky="ns")
        ttk.Button(nudge, text="X0 - PitchX", command=lambda: self.nudge_grid("x0", -1)).grid(row=0, column=0, padx=4, pady=3, sticky="ew")
        ttk.Button(nudge, text="X0 + PitchX", command=lambda: self.nudge_grid("x0", 1)).grid(row=0, column=1, padx=4, pady=3, sticky="ew")
        ttk.Button(nudge, text="Y0 - PitchY", command=lambda: self.nudge_grid("y0", -1)).grid(row=1, column=0, padx=4, pady=3, sticky="ew")
        ttk.Button(nudge, text="Y0 + PitchY", command=lambda: self.nudge_grid("y0", 1)).grid(row=1, column=1, padx=4, pady=3, sticky="ew")

        ttk.Label(controls, text="Zoom").grid(row=3, column=0, padx=4, pady=4, sticky="e")
        zoom_scale = ttk.Scale(
            controls,
            from_=0.5,
            to=8.0,
            variable=self.vars["zoom"],
            orient="horizontal",
            command=lambda _v: self.refresh_display(),
        )
        zoom_scale.grid(row=3, column=1, columnspan=4, sticky="ew", padx=4, pady=4)

        ttk.Checkbutton(
            controls,
            text="Mostrar predicciones",
            variable=self.show_predictions_var,
            command=self.refresh_display_and_text,
        ).grid(row=3, column=6, columnspan=2, sticky="w", padx=4, pady=4)

        btns = ttk.Frame(controls)
        btns.grid(row=3, column=8, columnspan=6, sticky="e", padx=4, pady=4)
        ttk.Button(btns, text="Abrir imagen", command=self.open_image).pack(side="left", padx=3)
        ttk.Button(btns, text="Actualizar grilla", command=self.apply_params).pack(side="left", padx=3)
        ttk.Button(btns, text="Llenar con predicciones", command=self.fill_text_from_predictions).pack(side="left", padx=3)
        ttk.Button(btns, text="Reentrenar desde TXT", command=self.retrain_from_reference).pack(side="left", padx=3)
        ttk.Button(btns, text="Guardar proyecto", command=self.save_project).pack(side="left", padx=3)
        ttk.Button(btns, text="Cargar proyecto", command=self.load_project).pack(side="left", padx=3)
        ttk.Button(btns, text="Exportar TXT", command=self.export_text).pack(side="left", padx=3)

        controls.columnconfigure(1, weight=1)
        controls.columnconfigure(4, weight=1)

        main = ttk.Panedwindow(top, orient="horizontal")
        main.pack(fill="both", expand=True, padx=8, pady=6)

        left = ttk.Frame(main)
        right = ttk.Frame(main, width=500)
        main.add(left, weight=4)
        main.add(right, weight=1)

        canvas_frame = ttk.Frame(left)
        canvas_frame.pack(fill="both", expand=True)
        self.canvas = tk.Canvas(canvas_frame, bg="#202020", highlightthickness=0)
        self.canvas.pack(side="left", fill="both", expand=True)
        vbar = ttk.Scrollbar(canvas_frame, orient="vertical", command=self.canvas.yview)
        vbar.pack(side="right", fill="y")
        hbar = ttk.Scrollbar(left, orient="horizontal", command=self.canvas.xview)
        hbar.pack(fill="x")
        self.canvas.configure(yscrollcommand=vbar.set, xscrollcommand=hbar.set)
        self.canvas.bind("<Button-1>", self.on_canvas_click)
        self.canvas.bind("<MouseWheel>", self.on_mousewheel)
        self.canvas.bind("<Shift-MouseWheel>", self.on_shift_mousewheel)

        preview_row = ttk.Frame(right)
        preview_row.pack(fill="x", pady=(0, 8))

        preview = ttk.LabelFrame(preview_row, text="Celda")
        preview.pack(side="left", fill="both", expand=True, padx=(0, 4))
        self.cell_label = ttk.Label(preview, anchor="center")
        self.cell_label.pack(fill="both", expand=True, padx=6, pady=6)

        predimg = ttk.LabelFrame(preview_row, text="Predicción")
        predimg.pack(side="left", fill="both", expand=True, padx=4)
        self.prediction_label = ttk.Label(predimg, anchor="center")
        self.prediction_label.pack(fill="both", expand=True, padx=6, pady=6)

        context = ttk.LabelFrame(preview_row, text="Contexto")
        context.pack(side="left", fill="both", expand=True, padx=(4, 0))
        self.context_label = ttk.Label(context, anchor="center")
        self.context_label.pack(fill="both", expand=True, padx=6, pady=6)

        pred = ttk.LabelFrame(right, text="Reconocimiento")
        pred.pack(fill="x", pady=(0, 8))
        ttk.Label(pred, textvariable=self.prediction_var).pack(anchor="w", padx=8, pady=(8, 2))
        self.template_status_label = ttk.Label(pred, text="")
        self.template_status_label.pack(anchor="w", padx=8, pady=(0, 2))
        ttk.Label(pred, text="Teclado: escribí el carácter correcto. Atajos: F1=,  F2=.  F3='  F4=espacio").pack(
            anchor="w", padx=8, pady=(2, 8)
        )

        textf = ttk.LabelFrame(right, text="Texto reconstruido")
        textf.pack(fill="both", expand=True)
        text_inner = ttk.Frame(textf)
        text_inner.pack(fill="both", expand=True)
        self.text_box = tk.Text(text_inner, wrap="none", width=56, height=24, font=("Consolas", 10), undo=False)
        self.text_box.grid(row=0, column=0, sticky="nsew")
        tscroll = ttk.Scrollbar(text_inner, orient="vertical", command=self.text_box.yview)
        tscroll.grid(row=0, column=1, sticky="ns")
        xscroll = ttk.Scrollbar(text_inner, orient="horizontal", command=self.text_box.xview)
        xscroll.grid(row=1, column=0, sticky="ew")
        self.text_box.configure(yscrollcommand=tscroll.set, xscrollcommand=xscroll.set)
        self.text_box.bind("<Button-1>", lambda e: self.focus_main())
        self.text_box.bind("<Key>", lambda e: "break")
        text_inner.columnconfigure(0, weight=1)
        text_inner.rowconfigure(0, weight=1)

        status = ttk.Frame(self.root)
        status.pack(fill="x", padx=8, pady=(0, 6))
        ttk.Label(status, textvariable=self.status_var).pack(side="left")
        ttk.Label(status, textvariable=self.image_size_var).pack(side="right")

    def focus_main(self) -> None:
        self.root.focus_force()
        self.canvas.focus_set()

    def _bind_keys(self) -> None:
        self.root.bind("<Left>", lambda e: self.move_selection(-1, 0))
        self.root.bind("<Right>", lambda e: self.move_selection(1, 0))
        self.root.bind("<Up>", lambda e: self.move_selection(0, -1))
        self.root.bind("<Down>", lambda e: self.move_selection(0, 1))
        self.root.bind("<Return>", lambda e: self.move_selection(-self.selected_cell[0], 1))
        self.root.bind("<BackSpace>", self.on_backspace)
        self.root.bind("<F1>", lambda e: self.accept_char(","))
        self.root.bind("<F2>", lambda e: self.accept_char("."))
        self.root.bind("<F3>", lambda e: self.accept_char("'"))
        self.root.bind("<F4>", lambda e: self.accept_char(" "))
        self.root.bind_all("<Key>", self.on_keypress)

    def update_template_status(self) -> None:
        ref_state = "TXT: sí" if self.reference_lines else "TXT: no"
        self.template_status_label.configure(
            text=f"Templates: espacio={self.template_store.count(' ')}  uno={self.template_store.count('1')}  conflictos={len(self.validation_map)}  {ref_state}"
        )

    def invalidate_predictions(self) -> None:
        self.prediction_cache.clear()

    def invalidate_prediction_for_cell(self, col: int, row: int) -> None:
        key = f"{row},{col}"
        self.prediction_cache.pop(key, None)

    def get_params(self) -> GridParams:
        return GridParams(
            x0=float(self.vars["x0"].get()),
            y0=float(self.vars["y0"].get()),
            cell_w=float(self.vars["cell_w"].get()),
            cell_h=float(self.vars["cell_h"].get()),
            pitch_x=float(self.vars["pitch_x"].get()),
            pitch_y=float(self.vars["pitch_y"].get()),
            drift_x=float(self.vars["drift_x"].get()),
            drift_y=float(self.vars["drift_y"].get()),
            cols=int(self.vars["cols"].get()),
            rows=int(self.vars["rows"].get()),
            threshold=int(self.vars["threshold"].get()),
            zoom=float(self.vars["zoom"].get()),
        )

    def refresh_display_and_text(self) -> None:
        if self.original_image is not None:
            self.refresh_display()
            self.update_selection_views()
        else:
            self.rebuild_text_box()

    def apply_params(self) -> None:
        self.params = self.get_params()
        self.invalidate_predictions()
        if self.original_image is not None:
            self.make_binary_image()
            self.refresh_display()
            self.update_selection_views()
            self.focus_main()

    def nudge_grid(self, axis: str, sign: int) -> None:
        step = float(self.vars["pitch_x"].get()) if axis == "x0" else float(self.vars["pitch_y"].get())
        current = float(self.vars[axis].get())
        self.vars[axis].set(current + (step * sign))
        self.apply_params()

    def open_image(self) -> None:
        path = filedialog.askopenfilename(
            title="Abrir imagen",
            filetypes=[("Imágenes", "*.png;*.jpg;*.jpeg;*.bmp;*.tif;*.tiff"), ("Todos", "*.*")],
        )
        if not path:
            return
        try:
            img = Image.open(path).convert("L")
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo abrir la imagen:\n{e}")
            return
        self.image_path = path
        self.original_image = img
        self.recognized = {}
        self.reference_lines = []
        self.validation_map = {}
        self.invalidate_predictions()
        self.selected_cell = (0, 0)
        self.apply_params()
        self.status_var.set(f"Imagen cargada: {os.path.basename(path)}")
        self.image_size_var.set(f"Imagen: {img.width} x {img.height}")
        self.rebuild_text_box()
        self.focus_main()

    def make_binary_image(self) -> None:
        if self.original_image is None:
            return
        thr = self.get_params().threshold
        img = ImageOps.autocontrast(self.original_image)
        self.binary_image = img.point(lambda p: 255 if p > thr else 0, mode="1").convert("L")

    def refresh_display(self) -> None:
        if self.original_image is None or self.binary_image is None:
            return
        self.params = self.get_params()
        zoom = max(0.1, self.params.zoom)
        base = self.original_image.convert("RGB")
        disp_w = max(1, int(round(base.width * zoom)))
        disp_h = max(1, int(round(base.height * zoom)))
        disp = base.resize((disp_w, disp_h), Image.Resampling.NEAREST)

        draw = ImageDraw.Draw(disp)

        for r in range(self.params.rows):
            for c in range(self.params.cols):
                x, y = self.cell_origin(c, r)
                x1 = x * zoom
                y1 = y * zoom
                x2 = (x + self.params.cell_w) * zoom
                y2 = (y + self.params.cell_h) * zoom

                key = f"{r},{c}"
                if (c, r) == self.selected_cell:
                    color = "#ff4040"
                elif key in self.validation_map:
                    color = "#ff00ff"
                else:
                    color = "#00ff88"
                width = 2 if key in self.validation_map or (c, r) == self.selected_cell else 1
                draw.rectangle([x1, y1, x2, y2], outline=color, width=width)

                if not self.show_predictions_var.get():
                    continue

                manual = self.recognized.get(key)
                label = None
                fill = None

                if manual is not None:
                    label = "␠" if manual == " " else manual
                    fill = "#ffffff"
                else:
                    pred, dist = self.predict_cell(c, r)
                    if pred is not None and dist is not None and dist < PREDICTION_MAX_DIST:
                        label = "␠" if pred == " " else pred
                        fill = "#00ff00"

                if label:
                    tx = int(round(x1 + 1))
                    ty = int(round(y1 + 0))
                    draw.text((tx, ty), label, fill=fill, font=self.overlay_font, stroke_width=1, stroke_fill="#002200")

        self.display_image = disp
        self.display_photo = ImageTk.PhotoImage(disp)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, image=self.display_photo, anchor="nw")
        self.canvas.configure(scrollregion=(0, 0, disp.width, disp.height))
        self.ensure_selection_visible()

    def cell_origin(self, col: int, row: int) -> Tuple[float, float]:
        p = self.params
        x = p.x0 + col * p.pitch_x + row * p.drift_x
        y = p.y0 + row * p.pitch_y + col * p.drift_y
        return x, y

    def cell_bbox(self, col: int, row: int) -> Tuple[int, int, int, int]:
        x, y = self.cell_origin(col, row)
        x1 = int(round(x))
        y1 = int(round(y))
        x2 = int(round(x + self.params.cell_w))
        y2 = int(round(y + self.params.cell_h))
        return x1, y1, x2, y2

    def crop_cell(self, col: int, row: int) -> Optional[Image.Image]:
        if self.binary_image is None:
            return None
        x1, y1, x2, y2 = self.cell_bbox(col, row)
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(self.binary_image.width, x2)
        y2 = min(self.binary_image.height, y2)
        if x2 <= x1 or y2 <= y1:
            return None
        return self.binary_image.crop((x1, y1, x2, y2))

    def make_signature(self, img: Image.Image) -> List[int]:
        norm = img.resize((16, 20), Image.Resampling.NEAREST)
        if hasattr(norm, "get_flattened_data"):
            data = list(norm.get_flattened_data())
        else:
            data = list(norm.getdata())
        return [1 if px < 128 else 0 for px in data]

    def predict_cell(self, col: int, row: int) -> Tuple[Optional[str], Optional[float]]:
        key = f"{row},{col}"
        if key in self.recognized:
            return self.recognized[key], 0.0
        if key in self.prediction_cache:
            return self.prediction_cache[key]

        crop = self.crop_cell(col, row)
        if crop is None:
            self.prediction_cache[key] = (None, None)
            return None, None

        sig = self.make_signature(crop)
        result = self.template_store.predict(sig)
        self.prediction_cache[key] = result
        return result

    def build_prediction_preview(self, pred: Optional[str], crop: Optional[Image.Image]) -> Image.Image:
        if crop is not None:
            target_w = max(1, crop.width * 8)
            target_h = max(1, crop.height * 8)
        else:
            target_w, target_h = 96, 128

        img = Image.new("RGB", (target_w, target_h), "white")
        draw = ImageDraw.Draw(img)
        draw.rectangle([0, 0, target_w - 1, target_h - 1], outline="#888888", width=1)

        if pred is None:
            text = "?"
        elif pred == " ":
            text = "␠"
        else:
            text = pred

        font_size = max(12, min(target_w, target_h) - 10)
        font = load_overlay_font(font_size)
        bbox = draw.textbbox((0, 0), text, font=font, stroke_width=1)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]

        while (tw > target_w - 12 or th > target_h - 12) and font_size > 10:
            font_size -= 2
            font = load_overlay_font(font_size)
            bbox = draw.textbbox((0, 0), text, font=font, stroke_width=1)
            tw = bbox[2] - bbox[0]
            th = bbox[3] - bbox[1]

        tx = (target_w - tw) // 2 - bbox[0]
        ty = (target_h - th) // 2 - bbox[1]
        draw.text((tx, ty), text, fill="#008800", font=font, stroke_width=1, stroke_fill="#ccffcc")
        return img

    def update_selection_views(self) -> None:
        col, row = self.selected_cell
        crop = self.crop_cell(col, row)
        if crop is None:
            return

        zoomed = crop.resize((max(1, crop.width * 8), max(1, crop.height * 8)), Image.Resampling.NEAREST)
        self.cell_photo = ImageTk.PhotoImage(zoomed)
        self.cell_label.configure(image=self.cell_photo)

        pred, dist = self.predict_cell(col, row)
        pred_img = self.build_prediction_preview(pred, crop)
        self.prediction_photo = ImageTk.PhotoImage(pred_img)
        self.prediction_label.configure(image=self.prediction_photo)

        if self.original_image is not None:
            x1, y1, x2, y2 = self.cell_bbox(col, row)
            pad = 8
            cx1 = max(0, x1 - pad)
            cy1 = max(0, y1 - pad)
            cx2 = min(self.original_image.width, x2 + pad)
            cy2 = min(self.original_image.height, y2 + pad)
            ctx = self.original_image.crop((cx1, cy1, cx2, cy2)).convert("RGB")
            dr = ImageDraw.Draw(ctx)
            dr.rectangle([x1 - cx1, y1 - cy1, x2 - cx1, y2 - cy1], outline="#ff2020", width=1)
            ctx = ctx.resize((zoomed.width, zoomed.height), Image.Resampling.NEAREST)
            self.context_photo = ImageTk.PhotoImage(ctx)
            self.context_label.configure(image=self.context_photo)

        expected = self.get_reference_char(col, row)
        if pred is None:
            base_msg = "Predicción: ?"
        else:
            shown = "espacio" if pred == " " else repr(pred)
            base_msg = f"Predicción: {shown}  dist={dist:.1f}"

        if expected is not None:
            exp_text = "espacio" if expected == " " else repr(expected)
            conflict = "  [NO COINCIDE]" if f"{row},{col}" in self.validation_map else "  [OK]"
            self.prediction_var.set(f"{base_msg}   TXT={exp_text}{conflict}")
        else:
            self.prediction_var.set(base_msg)

        self.rebuild_text_box(highlight=(col, row))
        self.update_template_status()

    def on_canvas_click(self, event: tk.Event) -> None:
        if self.display_image is None:
            return
        zoom = self.params.zoom
        x = self.canvas.canvasx(event.x) / zoom
        y = self.canvas.canvasy(event.y) / zoom
        best = None
        best_score = None
        for r in range(self.params.rows):
            for c in range(self.params.cols):
                ox, oy = self.cell_origin(c, r)
                score = abs(x - (ox + self.params.cell_w / 2.0)) + abs(y - (oy + self.params.cell_h / 2.0))
                if best_score is None or score < best_score:
                    best_score = score
                    best = (c, r)
        if best is not None:
            self.selected_cell = best
            self.refresh_display()
            self.update_selection_views()
            self.focus_main()

    def ensure_selection_visible(self) -> None:
        if self.display_image is None:
            return
        zoom = self.params.zoom
        col, row = self.selected_cell
        x, y = self.cell_origin(col, row)
        x1 = x * zoom
        y1 = y * zoom
        x2 = (x + self.params.cell_w) * zoom
        y2 = (y + self.params.cell_h) * zoom
        self.canvas.update_idletasks()
        cw = max(1, self.canvas.winfo_width())
        ch = max(1, self.canvas.winfo_height())
        vx1 = self.canvas.canvasx(0)
        vy1 = self.canvas.canvasy(0)
        vx2 = vx1 + cw
        vy2 = vy1 + ch
        if x1 < vx1:
            self.canvas.xview_moveto(max(0, x1 / max(1, self.display_image.width)))
        elif x2 > vx2:
            self.canvas.xview_moveto(max(0, (x2 - cw) / max(1, self.display_image.width)))
        if y1 < vy1:
            self.canvas.yview_moveto(max(0, y1 / max(1, self.display_image.height)))
        elif y2 > vy2:
            self.canvas.yview_moveto(max(0, (y2 - ch) / max(1, self.display_image.height)))

    def move_selection(self, dx: int, dy: int) -> None:
        c, r = self.selected_cell
        c = min(max(0, c + dx), self.params.cols - 1)
        r = min(max(0, r + dy), self.params.rows - 1)
        self.selected_cell = (c, r)
        self.refresh_display()
        self.update_selection_views()
        self.focus_main()

    def on_backspace(self, _event: tk.Event) -> None:
        key = f"{self.selected_cell[1]},{self.selected_cell[0]}"
        self.recognized.pop(key, None)
        self.invalidate_prediction_for_cell(self.selected_cell[0], self.selected_cell[1])
        self.validate_against_reference()
        self.refresh_display()
        self.update_selection_views()
        self.move_selection(-1, 0)

    def accept_char(self, ch: str) -> None:
        col, row = self.selected_cell
        key = f"{row},{col}"
        self.recognized[key] = ch
        crop = self.crop_cell(col, row)
        if crop is not None:
            sig = self.make_signature(crop)
            self.template_store.add_sample(ch, sig)
            self.invalidate_predictions()
        self.validate_against_reference()
        self.refresh_display()
        self.update_selection_views()
        self.move_selection(1, 0)

    def on_keypress(self, event: tk.Event) -> None:
        if not event.char:
            return
        widget = event.widget
        if widget in (self.text_box,):
            return
        if ord(event.char) < 32:
            return
        if len(event.char) == 1:
            self.accept_char(event.char)

    def char_for_cell(self, c: int, r: int) -> str:
        key = f"{r},{c}"
        ch = self.recognized.get(key)
        if ch is not None:
            return ch
        if self.show_predictions_var.get():
            pred, dist = self.predict_cell(c, r)
            if pred is not None and dist is not None and dist < PREDICTION_MAX_DIST:
                return pred
        return "·"

    def rebuild_text_box(self, highlight: Optional[Tuple[int, int]] = None) -> None:
        self.text_box.configure(state="normal")
        self.text_box.delete("1.0", "end")
        for r in range(self.params.rows):
            chars = [self.char_for_cell(c, r) for c in range(self.params.cols)]
            self.text_box.insert("end", "".join(chars) + "\n")

        self.text_box.tag_delete("selcell")
        self.text_box.tag_delete("mismatch")
        self.text_box.tag_configure("mismatch", background="#5a004a", foreground="#ffffff")

        for key in self.validation_map:
            r_str, c_str = key.split(",")
            r = int(r_str)
            c = int(c_str)
            if 0 <= r < self.params.rows and 0 <= c < self.params.cols:
                self.text_box.tag_add("mismatch", f"{r + 1}.{c}", f"{r + 1}.{c + 1}")

        if highlight is not None:
            col, row = highlight
            index1 = f"{row + 1}.{col}"
            index2 = f"{row + 1}.{col + 1}"
            self.text_box.tag_add("selcell", index1, index2)
            self.text_box.tag_configure("selcell", background="#ffd27f")
            self.text_box.see(index1)
        self.text_box.configure(state="disabled")

    def fill_text_from_predictions(self) -> None:
        filled = 0
        for r in range(self.params.rows):
            for c in range(self.params.cols):
                key = f"{r},{c}"
                if key in self.recognized:
                    continue
                pred, dist = self.predict_cell(c, r)
                if pred is not None and dist is not None and dist < PREDICTION_MAX_DIST:
                    self.recognized[key] = pred
                    filled += 1
        self.validate_against_reference()
        self.refresh_display()
        self.update_selection_views()
        self.status_var.set(f"Se cargaron {filled} caracteres desde las predicciones.")

    def export_text(self) -> None:
        path = filedialog.asksaveasfilename(
            title="Exportar TXT",
            defaultextension=".txt",
            filetypes=[("Texto", "*.txt"), ("Todos", "*.*")],
        )
        if not path:
            return
        text = self.text_box.get("1.0", "end-1c")
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
        self.status_var.set(f"Texto exportado: {os.path.basename(path)}")

    def save_project(self) -> None:
        path = filedialog.asksaveasfilename(
            title="Guardar proyecto",
            defaultextension=PROJECT_EXT,
            filetypes=[("Proyecto OCR", f"*{PROJECT_EXT}"), ("JSON", "*.json")],
        )
        if not path:
            return
        data = {
            "image_path": self.image_path,
            "params": asdict(self.get_params()),
            "recognized": self.recognized,
            "selected_cell": list(self.selected_cell),
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        self.status_var.set(f"Proyecto guardado: {os.path.basename(path)}")

    def project_txt_path(self, project_path: str) -> str:
        base, _ext = os.path.splitext(project_path)
        base, _ext = os.path.splitext(base)
        return base + ".txt"

    def load_reference_text_for_project(self, project_path: str) -> bool:
        txt_path = self.project_txt_path(project_path)
        print(txt_path)
        self.reference_lines = []
        if not os.path.exists(txt_path):
            return False
        try:
            with open(txt_path, "r", encoding="utf-8") as f:
                self.reference_lines = f.read().splitlines()
            return True
        except Exception:
            self.reference_lines = []
            return False

    def get_reference_char(self, col: int, row: int) -> Optional[str]:
        if row >= len(self.reference_lines):
            return None
        line = self.reference_lines[row]
        if col >= len(line):
            return None
        return line[col]

    def validate_against_reference(self) -> None:
        self.validation_map = {}
        if not self.reference_lines:
            self.update_template_status()
            return

        for r in range(self.params.rows):
            for c in range(self.params.cols):
                expected = self.get_reference_char(c, r)
                if expected is None:
                    continue
                predicted = self.char_for_cell(c, r)
                if predicted != expected:
                    self.validation_map[f"{r},{c}"] = True

        self.update_template_status()

    def retrain_from_reference(self) -> None:
        if not self.reference_lines:
            messagebox.showinfo("Reentrenar", "No hay TXT de referencia cargado.")
            return

        count = 0
        for r in range(self.params.rows):
            for c in range(self.params.cols):
                expected = self.get_reference_char(c, r)
                if expected is None:
                    continue
                crop = self.crop_cell(c, r)
                if crop is None:
                    continue
                sig = self.make_signature(crop)
                self.template_store.add_sample(expected, sig)
                count += 1

        self.invalidate_predictions()
        self.validate_against_reference()
        self.refresh_display()
        self.update_selection_views()
        self.status_var.set(f"Reentrenado con {count} muestras desde TXT.")

    def load_project(self) -> None:
        path = filedialog.askopenfilename(
            title="Cargar proyecto",
            filetypes=[("Proyecto OCR", f"*{PROJECT_EXT}"), ("JSON", "*.json"), ("Todos", "*.*")],
        )
        if not path:
            return
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        image_path = data.get("image_path")
        if image_path and os.path.exists(image_path):
            self.original_image = Image.open(image_path).convert("L")
            self.image_path = image_path
            self.image_size_var.set(f"Imagen: {self.original_image.width} x {self.original_image.height}")
        else:
            messagebox.showwarning("Aviso", "La imagen original no está disponible. Se cargará solo el proyecto.")

        params = data.get("params", {})
        for k, v in params.items():
            if k in self.vars:
                self.vars[k].set(v)

        self.recognized = {}
        self.validation_map = {}
        self.invalidate_predictions()

        sc = data.get("selected_cell", [0, 0])
        self.selected_cell = (int(sc[0]), int(sc[1]))

        txt_loaded = self.load_reference_text_for_project(path)

        if self.original_image is not None:
            self.apply_params()
            self.validate_against_reference()
            self.update_selection_views()

        self.rebuild_text_box()

        if txt_loaded:
            self.status_var.set(f"Proyecto cargado + TXT detectado: {os.path.basename(self.project_txt_path(path))}")
        else:
            self.status_var.set(f"Proyecto cargado: {os.path.basename(path)} (sin TXT asociado)")

    def on_mousewheel(self, event: tk.Event) -> None:
        delta = -1 if event.delta > 0 else 1
        self.canvas.yview_scroll(delta * 3, "units")

    def on_shift_mousewheel(self, event: tk.Event) -> None:
        delta = -1 if event.delta > 0 else 1
        self.canvas.xview_scroll(delta * 3, "units")

    def open_image_path(self, path: str) -> None:
        try:
            self.original_image = Image.open(path).convert("L")
            self.image_path = path
            self.image_size_var.set(f"Imagen: {self.original_image.width} x {self.original_image.height}")
            self.recognized = {}
            self.reference_lines = []
            self.validation_map = {}
            self.invalidate_predictions()
            self.apply_params()
            self.update_selection_views()
            self.status_var.set(f"Imagen cargada: {os.path.basename(path)}")
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo abrir la imagen:\n{e}")


if __name__ == "__main__":
    root = tk.Tk()
    style = ttk.Style()
    try:
        style.theme_use("clam")
    except Exception:
        pass
    app = OCRRefineApp(root)
    if len(sys.argv) > 1 and os.path.exists(sys.argv[1]):
        app.open_image_path(sys.argv[1])
    root.mainloop()
