from __future__ import annotations

import struct
import zlib
from dataclasses import dataclass
from pathlib import Path

from analysis import Config, TrialResult, data_range


FONT_5X7 = {
    " ": ["00000", "00000", "00000", "00000", "00000", "00000", "00000"],
    "-": ["00000", "00000", "00000", "11111", "00000", "00000", "00000"],
    ".": ["00000", "00000", "00000", "00000", "00000", "01100", "01100"],
    ":": ["00000", "01100", "01100", "00000", "01100", "01100", "00000"],
    "(": ["00110", "01100", "01100", "01100", "01100", "01100", "00110"],
    ")": ["01100", "00110", "00110", "00110", "00110", "00110", "01100"],
    ",": ["00000", "00000", "00000", "00000", "00110", "00110", "01100"],
    "/": ["00001", "00010", "00100", "01000", "10000", "00000", "00000"],
    "=": ["00000", "11111", "00000", "11111", "00000", "00000", "00000"],
    "0": ["01110", "10001", "10011", "10101", "11001", "10001", "01110"],
    "1": ["00100", "01100", "00100", "00100", "00100", "00100", "01110"],
    "2": ["01110", "10001", "00001", "00010", "00100", "01000", "11111"],
    "3": ["11110", "00001", "00001", "01110", "00001", "00001", "11110"],
    "4": ["00010", "00110", "01010", "10010", "11111", "00010", "00010"],
    "5": ["11111", "10000", "10000", "11110", "00001", "00001", "11110"],
    "6": ["01110", "10000", "10000", "11110", "10001", "10001", "01110"],
    "7": ["11111", "00001", "00010", "00100", "01000", "01000", "01000"],
    "8": ["01110", "10001", "10001", "01110", "10001", "10001", "01110"],
    "9": ["01110", "10001", "10001", "01111", "00001", "00001", "01110"],
    "A": ["01110", "10001", "10001", "11111", "10001", "10001", "10001"],
    "B": ["11110", "10001", "10001", "11110", "10001", "10001", "11110"],
    "C": ["01110", "10001", "10000", "10000", "10000", "10001", "01110"],
    "D": ["11110", "10001", "10001", "10001", "10001", "10001", "11110"],
    "E": ["11111", "10000", "10000", "11110", "10000", "10000", "11111"],
    "F": ["11111", "10000", "10000", "11110", "10000", "10000", "10000"],
    "G": ["01110", "10001", "10000", "10111", "10001", "10001", "01110"],
    "H": ["10001", "10001", "10001", "11111", "10001", "10001", "10001"],
    "I": ["01110", "00100", "00100", "00100", "00100", "00100", "01110"],
    "J": ["00001", "00001", "00001", "00001", "10001", "10001", "01110"],
    "K": ["10001", "10010", "10100", "11000", "10100", "10010", "10001"],
    "L": ["10000", "10000", "10000", "10000", "10000", "10000", "11111"],
    "M": ["10001", "11011", "10101", "10101", "10001", "10001", "10001"],
    "N": ["10001", "11001", "10101", "10011", "10001", "10001", "10001"],
    "O": ["01110", "10001", "10001", "10001", "10001", "10001", "01110"],
    "P": ["11110", "10001", "10001", "11110", "10000", "10000", "10000"],
    "Q": ["01110", "10001", "10001", "10001", "10101", "10010", "01101"],
    "R": ["11110", "10001", "10001", "11110", "10100", "10010", "10001"],
    "S": ["01111", "10000", "10000", "01110", "00001", "00001", "11110"],
    "T": ["11111", "00100", "00100", "00100", "00100", "00100", "00100"],
    "U": ["10001", "10001", "10001", "10001", "10001", "10001", "01110"],
    "V": ["10001", "10001", "10001", "10001", "01010", "01010", "00100"],
    "W": ["10001", "10001", "10001", "10101", "10101", "10101", "01010"],
    "X": ["10001", "10001", "01010", "00100", "01010", "10001", "10001"],
    "Y": ["10001", "10001", "01010", "00100", "00100", "00100", "00100"],
    "Z": ["11111", "00001", "00010", "00100", "01000", "10000", "11111"],
}


@dataclass
class PlotRegion:
    x: int
    y: int
    width: int
    height: int


class RasterCanvas:
    def __init__(self, width: int, height: int, background: tuple[int, int, int]) -> None:
        self.width = width
        self.height = height
        self.pixels = [bytearray(background * width) for _ in range(height)]

    def set_pixel(self, x: int, y: int, color: tuple[int, int, int]) -> None:
        if 0 <= x < self.width and 0 <= y < self.height:
            offset = x * 3
            row = self.pixels[y]
            row[offset] = color[0]
            row[offset + 1] = color[1]
            row[offset + 2] = color[2]

    def fill_rect(self, x: int, y: int, width: int, height: int, color: tuple[int, int, int]) -> None:
        x0 = max(0, x)
        y0 = max(0, y)
        x1 = min(self.width, x + width)
        y1 = min(self.height, y + height)
        for py in range(y0, y1):
            row = self.pixels[py]
            for px in range(x0, x1):
                offset = px * 3
                row[offset] = color[0]
                row[offset + 1] = color[1]
                row[offset + 2] = color[2]

    def draw_line(self, x0: int, y0: int, x1: int, y1: int, color: tuple[int, int, int], thickness: int = 1) -> None:
        dx = abs(x1 - x0)
        dy = -abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx + dy
        while True:
            self.fill_rect(x0 - thickness // 2, y0 - thickness // 2, thickness, thickness, color)
            if x0 == x1 and y0 == y1:
                break
            e2 = 2 * err
            if e2 >= dy:
                err += dy
                x0 += sx
            if e2 <= dx:
                err += dx
                y0 += sy

    def draw_polyline(self, points: list[tuple[int, int]], color: tuple[int, int, int], thickness: int = 1) -> None:
        for start, end in zip(points, points[1:]):
            self.draw_line(start[0], start[1], end[0], end[1], color, thickness)

    def fill_circle(self, cx: int, cy: int, radius: int, color: tuple[int, int, int]) -> None:
        radius_sq = radius * radius
        for py in range(cy - radius, cy + radius + 1):
            for px in range(cx - radius, cx + radius + 1):
                if (px - cx) ** 2 + (py - cy) ** 2 <= radius_sq:
                    self.set_pixel(px, py, color)

    def draw_text(self, x: int, y: int, text: str, color: tuple[int, int, int], scale: int = 2) -> None:
        cursor_x = x
        for character in text.upper():
            glyph = FONT_5X7.get(character, FONT_5X7[" "])
            for row_index, row in enumerate(glyph):
                for column_index, bit in enumerate(row):
                    if bit == "1":
                        self.fill_rect(cursor_x + column_index * scale, y + row_index * scale, scale, scale, color)
            cursor_x += 6 * scale

    def draw_wrapped_text(
        self,
        x: int,
        y: int,
        text: str,
        max_width: int,
        color: tuple[int, int, int],
        scale: int = 2,
    ) -> int:
        max_chars = max(1, max_width // (6 * scale))
        words = text.upper().split()
        lines = []
        line = ""
        for word in words:
            candidate = word if not line else f"{line} {word}"
            if len(candidate) <= max_chars:
                line = candidate
            else:
                if line:
                    lines.append(line)
                line = word
        if line:
            lines.append(line)
        for index, line_text in enumerate(lines):
            self.draw_text(x, y + index * 9 * scale, line_text, color, scale)
        return len(lines) * 9 * scale


def map_x(value: float, plot: PlotRegion, x_min: float, x_max: float) -> int:
    return plot.x + round((value - x_min) / (x_max - x_min) * plot.width)


def map_y(value: float, plot: PlotRegion, y_min: float, y_max: float) -> int:
    return plot.y + plot.height - round((value - y_min) / (y_max - y_min) * plot.height)


def write_png(path: Path, canvas: RasterCanvas) -> None:
    raw = bytearray()
    for row in canvas.pixels:
        raw.append(0)
        raw.extend(row)

    def chunk(chunk_type: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + chunk_type
            + data
            + struct.pack(">I", zlib.crc32(chunk_type + data) & 0xFFFFFFFF)
        )

    ihdr = struct.pack(">IIBBBBB", canvas.width, canvas.height, 8, 2, 0, 0, 0)
    png_bytes = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", zlib.compress(bytes(raw), 9)) + chunk(b"IEND", b"")
    path.write_bytes(png_bytes)


def draw_panel(canvas: RasterCanvas, x: int, y: int, width: int, height: int, title: str) -> PlotRegion:
    canvas.fill_rect(x, y, width, height, (255, 255, 255))
    canvas.draw_line(x, y, x + width, y, (214, 214, 214))
    canvas.draw_line(x, y + height, x + width, y + height, (214, 214, 214))
    canvas.draw_line(x, y, x, y + height, (214, 214, 214))
    canvas.draw_line(x + width, y, x + width, y + height, (214, 214, 214))
    title_height = canvas.draw_wrapped_text(x + 18, y + 16, title, width - 36, (24, 24, 24), scale=2)
    return PlotRegion(x + 68, y + title_height + 30, width - 96, height - title_height - 64)


def draw_axes(
    canvas: RasterCanvas,
    plot: PlotRegion,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    x_label: str,
    y_label: str,
    x_ticks: list[float],
    y_ticks: list[float],
) -> None:
    canvas.fill_rect(plot.x, plot.y, plot.width, plot.height, (249, 249, 249))
    for tick in x_ticks:
        px = map_x(tick, plot, x_min, x_max)
        canvas.draw_line(px, plot.y, px, plot.y + plot.height, (234, 234, 234))
        canvas.draw_text(px - 12, plot.y + plot.height + 10, f"{tick:.1f}", (70, 70, 70), scale=1)
    for tick in y_ticks:
        py = map_y(tick, plot, y_min, y_max)
        canvas.draw_line(plot.x, py, plot.x + plot.width, py, (234, 234, 234))
        canvas.draw_text(plot.x - 42, py - 4, f"{tick:.1f}", (70, 70, 70), scale=1)
    canvas.draw_line(plot.x, plot.y + plot.height, plot.x + plot.width, plot.y + plot.height, (50, 50, 50))
    canvas.draw_line(plot.x, plot.y, plot.x, plot.y + plot.height, (50, 50, 50))
    canvas.draw_text(plot.x + plot.width // 2 - 40, plot.y + plot.height + 28, x_label, (30, 30, 30), scale=1)
    canvas.draw_text(plot.x, plot.y - 16, y_label, (30, 30, 30), scale=1)


def draw_curve_panel(
    canvas: RasterCanvas,
    plot: PlotRegion,
    x_values: list[float],
    point_values: list[float],
    curves: list[tuple[list[float], tuple[int, int, int]]],
    caption: str,
) -> None:
    x_min, x_max = data_range(x_values, 0.0)
    y_values = point_values[:]
    for curve_values, _ in curves:
        y_values.extend(curve_values)
    y_min, y_max = data_range(y_values)
    draw_axes(canvas, plot, x_min, x_max, y_min, y_max, "X (A.U.)", "Y (A.U.)", [-2, -1, 0, 1, 2], [round(y_min, 1), 0.0, round(y_max, 1)])
    for x_value, y_value in zip(x_values, point_values):
        canvas.fill_circle(map_x(x_value, plot, x_min, x_max), map_y(y_value, plot, y_min, y_max), 2, (31, 119, 180))
    for curve_values, color in curves:
        points = [(map_x(x, plot, x_min, x_max), map_y(y, plot, y_min, y_max)) for x, y in zip(x_values, curve_values)]
        canvas.draw_polyline(points, color, thickness=2)
    canvas.draw_wrapped_text(plot.x + 8, plot.y + 8, caption, plot.width - 16, (60, 60, 60), scale=1)


def draw_history_panel(
    canvas: RasterCanvas,
    plot: PlotRegion,
    trials: list[TrialResult],
    colors: dict[str, tuple[int, int, int]],
) -> None:
    epochs = [epoch for epoch, _, _ in trials[0].loss_history]
    y_values = []
    for trial in trials:
        y_values.extend([loss for _, loss, _ in trial.loss_history])
    y_min, y_max = data_range(y_values, 0.1)
    draw_axes(
        canvas,
        plot,
        float(min(epochs)),
        float(max(epochs)),
        y_min,
        y_max,
        "EPOCH",
        "DATA LOSS",
        [0.0, float(max(epochs) // 2), float(max(epochs))],
        [round(y_min, 1), round((y_min + y_max) / 2, 1), round(y_max, 1)],
    )
    for trial in trials:
        points = [
            (map_x(float(epoch), plot, float(min(epochs)), float(max(epochs))), map_y(loss, plot, y_min, y_max))
            for epoch, loss, _ in trial.loss_history
        ]
        canvas.draw_polyline(points, colors[trial.name], thickness=2)
    canvas.draw_wrapped_text(plot.x + 8, plot.y + 8, "BLACK = NO REG. RED = LASSO. GREEN = RIDGE.", plot.width - 16, (60, 60, 60), scale=1)


def draw_coefficients_panel(
    canvas: RasterCanvas,
    plot: PlotRegion,
    trials: list[TrialResult],
    colors: dict[str, tuple[int, int, int]],
    threshold: float,
) -> None:
    coeff_values = []
    for trial in trials:
        coeff_values.extend([abs(value) for value in trial.theta_final])
    y_min, y_max = 0.0, max(coeff_values) * 1.1 if coeff_values else 1.0
    total_coeffs = len(trials[0].theta_final)
    draw_axes(
        canvas,
        plot,
        0.0,
        float(total_coeffs + 1),
        y_min,
        y_max,
        "COEFFICIENT INDEX",
        "ABS VALUE",
        [1.0, float(total_coeffs // 2), float(total_coeffs)],
        [0.0, round(y_max / 2, 1), round(y_max, 1)],
    )
    offset_lookup = {"none": -10, "lasso": 0, "ridge": 10}
    for trial in trials:
        for index, value in enumerate(trial.theta_final, start=1):
            px = map_x(float(index), plot, 0.0, float(total_coeffs + 1)) + offset_lookup[trial.name]
            py = map_y(abs(value), plot, y_min, y_max)
            canvas.fill_rect(px - 4, py, 8, plot.y + plot.height - py, colors[trial.name])
    threshold_y = map_y(threshold, plot, y_min, y_max)
    canvas.draw_line(plot.x, threshold_y, plot.x + plot.width, threshold_y, (100, 100, 100), thickness=1)
    canvas.draw_wrapped_text(plot.x + 8, plot.y + 8, f"COUNT COEFFICIENTS ABOVE THRESHOLD = {threshold:.3f}.", plot.width - 16, (60, 60, 60), scale=1)


def draw_dimension_panel(
    canvas: RasterCanvas,
    plot: PlotRegion,
    cfg: Config,
    trials: list[TrialResult],
    colors: dict[str, tuple[int, int, int]],
) -> None:
    y_max = float(max(max(trial.dimension_estimate for trial in trials), 2) + 1)
    draw_axes(
        canvas,
        plot,
        0.0,
        4.0,
        0.0,
        y_max,
        "METHOD",
        "NONZERO COUNT",
        [1.0, 2.0, 3.0],
        [0.0, round(y_max / 2), round(y_max)],
    )
    for index, trial in enumerate(trials, start=1):
        px = map_x(float(index), plot, 0.0, 4.0)
        py = map_y(float(trial.dimension_estimate), plot, 0.0, y_max)
        canvas.fill_rect(px - 26, py, 52, plot.y + plot.height - py, colors[trial.name])
        canvas.draw_text(px - 18, plot.y + plot.height + 12, trial.name.upper(), (45, 45, 45), scale=1)
        canvas.draw_text(px - 6, py - 16, str(trial.dimension_estimate), (20, 20, 20), scale=1)
    true_y = map_y(2.0, plot, 0.0, y_max)
    canvas.draw_line(plot.x, true_y, plot.x + plot.width, true_y, (20, 20, 20), thickness=1)
    canvas.draw_wrapped_text(
        plot.x + 8,
        plot.y + 8,
        (
            f"FINAL STEP: COUNT RAW NON-ZERO COEFFICIENTS. TRUE DIMENSION = 2, "
            f"BUT THE THREE ESTIMATES ARE {trials[0].dimension_estimate}, "
            f"{trials[1].dimension_estimate}, AND {trials[2].dimension_estimate}."
        ),
        plot.width - 16,
        (60, 60, 60),
        scale=1,
    )


def render_regularization_figure(
    cfg: Config,
    x_values: list[float],
    y_values: list[float],
    y_clean: list[float],
    y_random: list[float],
    trials: list[TrialResult],
    random_loss: float,
) -> RasterCanvas:
    colors = {
        "none": (30, 30, 30),
        "lasso": (214, 39, 40),
        "ridge": (31, 119, 180),
    }
    canvas = RasterCanvas(1840, 1340, (243, 240, 233))
    canvas.draw_wrapped_text(
        40,
        26,
        "Raw Non-Zero Counts Overestimate Dimension In This Overparameterized Model",
        1750,
        (18, 18, 18),
        scale=3,
    )
    canvas.draw_wrapped_text(
        40,
        88,
        f"True dimension = 2. Model dimension = {2 * cfg.n_terms}. Gaussian noise is added to the data. Final estimate = raw non-zero coefficient count after training.",
        1750,
        (70, 70, 70),
        scale=1,
    )

    panel_width = 560
    panel_height = 510
    gap_x = 36
    gap_y = 34
    x0 = 40
    y0 = 138

    titles = [
        "1. Ground-Truth Data: The synthetic target uses exactly two coefficients",
        "2. Shared Random Start: All three optimizers begin from the same random 12-parameter model",
        "3. Fitted Curves: Compare no regularization, lasso, and ridge",
        "4. Training Loss: All methods reduce the data fit error",
        "5. Final Coefficients: Lasso drives most redundant coefficients to zero",
        "6. Dimensionality Estimate: Simply count the non-zero coefficients",
    ]
    plots = []
    for row in range(2):
        for col in range(3):
            plots.append(
                draw_panel(
                    canvas,
                    x0 + col * (panel_width + gap_x),
                    y0 + row * (panel_height + gap_y),
                    panel_width,
                    panel_height,
                    titles[row * 3 + col],
                )
            )

    draw_curve_panel(
        canvas,
        plots[0],
        x_values,
        y_values,
        [(y_clean, (44, 160, 44))],
        "BLUE = NOISY DATA. GREEN = TRUE CURVE. TRUE DIMENSION = 2.",
    )
    draw_curve_panel(
        canvas,
        plots[1],
        x_values,
        y_values,
        [(y_random, (255, 127, 14))],
        f"ORANGE = SHARED RANDOM MODEL. LOSS = {random_loss:.2E}. MODEL DIMENSION = {2 * cfg.n_terms}.",
    )
    draw_curve_panel(
        canvas,
        plots[2],
        x_values,
        y_values,
        [
            (y_clean, (44, 160, 44)),
            (trials[0].curve, colors["none"]),
            (trials[1].curve, colors["lasso"]),
            (trials[2].curve, colors["ridge"]),
        ],
        "GREEN = TRUTH. BLACK = NO REG. RED = LASSO. BLUE = RIDGE.",
    )
    draw_history_panel(canvas, plots[3], trials, colors)
    draw_coefficients_panel(canvas, plots[4], trials, colors, cfg.zero_threshold)
    draw_dimension_panel(canvas, plots[5], cfg, trials, colors)
    return canvas
