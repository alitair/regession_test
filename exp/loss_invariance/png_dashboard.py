from __future__ import annotations

import math
import struct
import zlib
from dataclasses import dataclass
from pathlib import Path

from analysis import Config, Summary, data_range


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
    "*": ["00000", "10001", "01010", "00100", "01010", "10001", "00000"],
    "+": ["00000", "00100", "00100", "11111", "00100", "00100", "00000"],
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


def color_mix(a: tuple[int, int, int], b: tuple[int, int, int], t: float) -> tuple[int, int, int]:
    return (
        round(a[0] + (b[0] - a[0]) * t),
        round(a[1] + (b[1] - a[1]) * t),
        round(a[2] + (b[2] - a[2]) * t),
    )


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
        canvas.draw_text(plot.x - 48, py - 4, f"{tick:.1f}", (70, 70, 70), scale=1)
    canvas.draw_line(plot.x, plot.y + plot.height, plot.x + plot.width, plot.y + plot.height, (50, 50, 50))
    canvas.draw_line(plot.x, plot.y, plot.x, plot.y + plot.height, (50, 50, 50))
    canvas.draw_text(plot.x + plot.width // 2 - 42, plot.y + plot.height + 28, x_label, (30, 30, 30), scale=1)
    canvas.draw_text(plot.x, plot.y - 16, y_label, (30, 30, 30), scale=1)


def draw_box(canvas: RasterCanvas, x: int, y: int, width: int, height: int, fill: tuple[int, int, int], border: tuple[int, int, int], text: str) -> None:
    canvas.fill_rect(x, y, width, height, fill)
    canvas.draw_line(x, y, x + width, y, border)
    canvas.draw_line(x, y + height, x + width, y + height, border)
    canvas.draw_line(x, y, x, y + height, border)
    canvas.draw_line(x + width, y, x + width, y + height, border)
    canvas.draw_wrapped_text(x + 10, y + 10, text, width - 20, (25, 25, 25), scale=1)


def draw_arrow(canvas: RasterCanvas, x0: int, y0: int, x1: int, y1: int, color: tuple[int, int, int]) -> None:
    canvas.draw_line(x0, y0, x1, y1, color, thickness=2)
    if abs(x1 - x0) >= abs(y1 - y0):
        direction = 1 if x1 >= x0 else -1
        canvas.draw_line(x1, y1, x1 - 8 * direction, y1 - 5, color, thickness=2)
        canvas.draw_line(x1, y1, x1 - 8 * direction, y1 + 5, color, thickness=2)
    else:
        direction = 1 if y1 >= y0 else -1
        canvas.draw_line(x1, y1, x1 - 5, y1 - 8 * direction, color, thickness=2)
        canvas.draw_line(x1, y1, x1 + 5, y1 - 8 * direction, color, thickness=2)


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


def draw_curve_panel(
    canvas: RasterCanvas,
    plot: PlotRegion,
    x_values: list[float],
    y_points: list[float],
    curves: list[tuple[list[float], tuple[int, int, int]]],
    x_label: str,
    y_label: str,
    caption: str,
) -> None:
    x_min, x_max = data_range(x_values, 0.0)
    combined = y_points[:]
    for curve_values, _ in curves:
        combined.extend(curve_values)
    y_min, y_max = data_range(combined)
    draw_axes(canvas, plot, x_min, x_max, y_min, y_max, x_label, y_label, [-2, -1, 0, 1, 2], [round(y_min, 1), 0.0, round(y_max, 1)])
    for x_value, y_value in zip(x_values, y_points):
        canvas.fill_circle(map_x(x_value, plot, x_min, x_max), map_y(y_value, plot, y_min, y_max), 2, (31, 119, 180))
    for curve_values, color in curves:
        points = [(map_x(x, plot, x_min, x_max), map_y(y, plot, y_min, y_max)) for x, y in zip(x_values, curve_values)]
        canvas.draw_polyline(points, color, thickness=2)
    canvas.draw_wrapped_text(plot.x + 6, plot.y + 8, caption, plot.width - 12, (60, 60, 60), scale=2)


def draw_spectrum_panel(canvas: RasterCanvas, plot: PlotRegion, eigenvalues: list[float], active_rank: int, nullity: int) -> None:
    spectrum = [math.log10(max(value, 1e-15)) for value in eigenvalues]
    x_min, x_max = 1.0, float(len(eigenvalues))
    y_min, y_max = -15.0, max(spectrum) + 0.4
    draw_axes(
        canvas,
        plot,
        x_min,
        x_max,
        y_min,
        y_max,
        "DIRECTION INDEX (A.U.)",
        "LOG10 EIGENVALUE",
        [1, len(eigenvalues) // 2, len(eigenvalues)],
        [-15, -10, -5, 0, round(y_max, 1)],
    )
    bar_width = max(4, plot.width // (len(eigenvalues) * 2))
    for index, value in enumerate(spectrum, start=1):
        px = map_x(float(index), plot, x_min, x_max)
        py = map_y(value, plot, y_min, y_max)
        color = (130, 130, 130) if index <= nullity else (127, 60, 141)
        canvas.fill_rect(px - bar_width, py, 2 * bar_width, plot.y + plot.height - py, color)
    canvas.draw_wrapped_text(
        plot.x + 12,
        plot.y + 8,
        f"COUNT EIGENVALUES ABOVE TOLERANCE. HERE {active_rank} MODES ARE ACTIVE AND {nullity} MODES ARE FLAT.",
        plot.width - 20,
        (60, 60, 60),
        scale=1,
    )


def draw_projection_panel(
    canvas: RasterCanvas,
    plot: PlotRegion,
    null_points: list[tuple[float, float]],
    aggregate_points: list[tuple[float, float]],
    true_point: tuple[float, float],
    fitted_point: tuple[float, float],
) -> None:
    left = PlotRegion(plot.x, plot.y + 24, plot.width // 2 - 20, plot.height - 40)
    right = PlotRegion(plot.x + plot.width // 2 + 20, plot.y + 24, plot.width // 2 - 20, plot.height - 40)

    null_x = [point[0] for point in null_points]
    null_y = [point[1] for point in null_points]
    null_x_min, null_x_max = data_range(null_x)
    null_y_min, null_y_max = data_range(null_y)
    draw_axes(
        canvas,
        left,
        null_x_min,
        null_x_max,
        null_y_min,
        null_y_max,
        "U = A1-A2 (A.U.)",
        "V = B1-B2 (A.U.)",
        [round(null_x_min, 1), 0.0, round(null_x_max, 1)],
        [round(null_y_min, 1), 0.0, round(null_y_max, 1)],
    )
    for x_value, y_value in null_points:
        canvas.fill_circle(map_x(x_value, left, null_x_min, null_x_max), map_y(y_value, left, null_y_min, null_y_max), 3, (214, 39, 40))

    agg_x = [point[0] for point in aggregate_points] + [true_point[0], fitted_point[0]]
    agg_y = [point[1] for point in aggregate_points] + [true_point[1], fitted_point[1]]
    agg_x_min, agg_x_max = data_range(agg_x)
    agg_y_min, agg_y_max = data_range(agg_y)
    draw_axes(
        canvas,
        right,
        agg_x_min,
        agg_x_max,
        agg_y_min,
        agg_y_max,
        "A = SUM A_N (A.U.)",
        "B = SUM B_N (A.U.)",
        [round(agg_x_min, 1), round(fitted_point[0], 1), round(agg_x_max, 1)],
        [round(agg_y_min, 1), round(fitted_point[1], 1), round(agg_y_max, 1)],
    )
    for x_value, y_value in aggregate_points:
        canvas.fill_circle(map_x(x_value, right, agg_x_min, agg_x_max), map_y(y_value, right, agg_y_min, agg_y_max), 3, (255, 204, 0))
    canvas.fill_circle(map_x(true_point[0], right, agg_x_min, agg_x_max), map_y(true_point[1], right, agg_y_min, agg_y_max), 6, (255, 255, 255))
    canvas.draw_line(
        map_x(true_point[0], right, agg_x_min, agg_x_max) - 7,
        map_y(true_point[1], right, agg_y_min, agg_y_max),
        map_x(true_point[0], right, agg_x_min, agg_x_max) + 7,
        map_y(true_point[1], right, agg_y_min, agg_y_max),
        (0, 0, 0),
    )
    canvas.draw_line(
        map_x(true_point[0], right, agg_x_min, agg_x_max),
        map_y(true_point[1], right, agg_y_min, agg_y_max) - 7,
        map_x(true_point[0], right, agg_x_min, agg_x_max),
        map_y(true_point[1], right, agg_y_min, agg_y_max) + 7,
        (0, 0, 0),
    )
    canvas.draw_wrapped_text(
        plot.x,
        plot.y,
        "LEFT: EQUAL-LOSS SAMPLES MOVE FREELY IN NULL DIRECTIONS. RIGHT: THE SAME SAMPLES COLLAPSE TO ONE AGGREGATE POINT THAT FIXES THE FUNCTION.",
        plot.width,
        (60, 60, 60),
        scale=1,
    )


def render_process_figure(
    cfg: Config,
    summary: Summary,
    x_values: list[float],
    y_values: list[float],
    y_clean: list[float],
    y_random: list[float],
    y_fit: list[float],
    random_loss: float,
    all_eigenvalues: list[float],
    null_projection_points: list[tuple[float, float]],
    aggregate_projection_points: list[tuple[float, float]],
) -> RasterCanvas:
    canvas = RasterCanvas(1840, 1340, (243, 240, 233))
    canvas.draw_wrapped_text(
        44,
        26,
        "Only Two Directions Matter For The Loss In This Toy Model",
        1750,
        (18, 18, 18),
        scale=3,
    )
    canvas.draw_wrapped_text(
        44,
        90,
        f"Ground-truth dimension = 2. Overparameterized model dimension = {2 * cfg.n_terms}. Equal-loss models sampled = {len(aggregate_projection_points)}.",
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

    panel_titles = [
        "1. Ground-Truth Data: The dataset lies on a quadratic curve",
        "2. Untrained Random Model: Random parameters do not match the data",
        "3. Fit The Overparameterized Model: Training recovers the target function",
        "4. Generate Equal-Loss Samples: Change parameters while keeping the same function",
        "5. Recover Dimensionality: Count the active Hessian directions",
        "6. Many Parameters, One Function: Null directions move in parameter space, not function space",
    ]
    p1 = draw_panel(canvas, x0 + 0 * (panel_width + gap_x), y0 + 0 * (panel_height + gap_y), panel_width, panel_height, panel_titles[0])
    p2 = draw_panel(canvas, x0 + 1 * (panel_width + gap_x), y0 + 0 * (panel_height + gap_y), panel_width, panel_height, panel_titles[1])
    p3 = draw_panel(canvas, x0 + 2 * (panel_width + gap_x), y0 + 0 * (panel_height + gap_y), panel_width, panel_height, panel_titles[2])
    p4 = draw_panel(canvas, x0 + 0 * (panel_width + gap_x), y0 + 1 * (panel_height + gap_y), panel_width, panel_height, panel_titles[3])
    p5 = draw_panel(canvas, x0 + 1 * (panel_width + gap_x), y0 + 1 * (panel_height + gap_y), panel_width, panel_height, panel_titles[4])
    p6 = draw_panel(canvas, x0 + 2 * (panel_width + gap_x), y0 + 1 * (panel_height + gap_y), panel_width, panel_height, panel_titles[5])

    draw_curve_panel(
        canvas,
        p1,
        x_values,
        y_values,
        [(y_clean, (44, 160, 44))],
        "X (A.U.)",
        "Y (A.U.)",
        "BLUE DOTS = DATA. GREEN CURVE = TRUE GENERATOR. TRUE DIMENSION = 2.",
    )

    draw_curve_panel(
        canvas,
        p2,
        x_values,
        y_values,
        [(y_random, (255, 127, 14))],
        "X (A.U.)",
        "Y (A.U.)",
        f"ORANGE = RANDOM MODEL IN A {2 * cfg.n_terms}-DIMENSION PARAMETER SPACE. LOSS = {random_loss:.2E}.",
    )

    draw_curve_panel(
        canvas,
        p3,
        x_values,
        y_values,
        [(y_clean, (44, 160, 44)), (y_random, (255, 127, 14)), (y_fit, (214, 39, 40))],
        "X (A.U.)",
        "Y (A.U.)",
        f"RED = FITTED MODEL. GREEN = TRUTH. OVERPARAMETERIZED DIMENSION = {2 * cfg.n_terms}.",
    )

    flow_x = p4.x + 4
    flow_y = p4.y + 44
    box_w = p4.width - 8
    box_h = 54
    draw_box(canvas, flow_x, flow_y, box_w, box_h, (235, 246, 255), (90, 145, 200), "START FROM A FITTED SOLUTION THETA*")
    draw_box(canvas, flow_x, flow_y + 82, box_w, box_h + 12, (241, 250, 232), (110, 160, 90), "DRAW DELTA_A AND DELTA_B SO THAT SUM DELTA_A = 0 AND SUM DELTA_B = 0")
    draw_box(canvas, flow_x, flow_y + 178, box_w, box_h + 12, (255, 245, 225), (200, 145, 70), "FORM A NEW PARAMETER VECTOR THETA_PRIME = THETA* + DELTA_THETA")
    draw_box(canvas, flow_x, flow_y + 274, box_w, box_h + 12, (245, 236, 255), (145, 110, 190), "THE AGGREGATE COEFFICIENTS A = SUM A_N AND B = SUM B_N STAY FIXED")
    draw_box(canvas, flow_x, flow_y + 370, box_w, box_h + 12, (255, 234, 236), (190, 90, 110), "THE PREDICTION YHAT(X) AND THE LOSS ARE UNCHANGED, SO THETA_PRIME IS ANOTHER EQUAL-LOSS SAMPLE")
    draw_arrow(canvas, flow_x + box_w // 2, flow_y + box_h, flow_x + box_w // 2, flow_y + 82, (85, 85, 85))
    draw_arrow(canvas, flow_x + box_w // 2, flow_y + 148, flow_x + box_w // 2, flow_y + 178, (85, 85, 85))
    draw_arrow(canvas, flow_x + box_w // 2, flow_y + 244, flow_x + box_w // 2, flow_y + 274, (85, 85, 85))
    draw_arrow(canvas, flow_x + box_w // 2, flow_y + 340, flow_x + box_w // 2, flow_y + 370, (85, 85, 85))

    spectrum_text_box = PlotRegion(p5.x, p5.y, p5.width, 118)
    draw_box(
        canvas,
        spectrum_text_box.x,
        spectrum_text_box.y,
        spectrum_text_box.width,
        spectrum_text_box.height,
        (245, 245, 255),
        (132, 110, 190),
        f"BUILD THE HESSIAN AT THE FITTED SOLUTION, COMPUTE ITS EIGENVALUES, AND COUNT HOW MANY ARE LARGER THAN A SMALL TOLERANCE. HERE THE ACTIVE COUNT IS {summary.active_rank}, WHICH MATCHES THE TRUE DATA DIMENSION OF 2.",
    )
    spectrum_plot = PlotRegion(p5.x, p5.y + 146, p5.width, p5.height - 150)
    draw_spectrum_panel(canvas, spectrum_plot, all_eigenvalues, summary.active_rank, summary.nullity)

    draw_projection_panel(
        canvas,
        p6,
        null_projection_points,
        aggregate_projection_points,
        (cfg.m1, cfg.m2),
        (summary.a_fit, summary.b_fit),
    )
    canvas.draw_wrapped_text(
        p6.x,
        p6.y + p6.height - 26,
        f"THE KEY IDEA IS THAT {len(aggregate_projection_points)} EQUAL-LOSS MODELS CAN EXIST IN A {2 * cfg.n_terms}-DIMENSION PARAMETER SPACE, BUT THE LOSS ONLY RESPONDS TO TWO AGGREGATED DIRECTIONS. THAT IS WHY THE RECOVERED DIMENSION IS 2.",
        p6.width,
        (60, 60, 60),
        scale=1,
    )

    return canvas
