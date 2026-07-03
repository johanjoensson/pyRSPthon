"""
Console report formatting for the runs SCF driver.

All user-facing layout lives here: TTY detection, ANSI color, unicode/ASCII
glyph fallbacks, and the header/iteration/summary blocks. Block builders
return (text, color) pairs; the color is applied only by the console log
handler, so the runs.info log file never receives escape codes.
"""

import logging
import math
import os
import sys

# Log level between INFO and WARNING for the final run summary, so that
# quiet mode (console level = SUMMARY) still reports how the run ended.
SUMMARY = 25
logging.addLevelName(SUMMARY, "SUMMARY")

WIDTH = 62

_ANSI = {
    "green": "\033[32m",
    "yellow": "\033[33m",
    "red": "\033[31m",
    "dim": "\033[2m",
    "bold": "\033[1m",
}
_RESET = "\033[0m"


def detect_style(stream=None) -> bool:
    """True when stream (default stdout) can take ANSI color and unicode."""
    stream = sys.stdout if stream is None else stream
    if os.environ.get("NO_COLOR"):
        return False
    if os.environ.get("TERM") == "dumb":
        return False
    return hasattr(stream, "isatty") and stream.isatty()


def colorize(text: str, color) -> str:
    code = _ANSI.get(color)
    return f"{code}{text}{_RESET}" if code else text


class ConsoleFormatter(logging.Formatter):
    """
    Apply per-record color hints (logger.log(..., extra={"color": ...})) on
    styled consoles; warnings and errors get level colors by default.
    """

    def __init__(self, styled: bool):
        super().__init__("%(message)s")
        self.styled = styled

    def format(self, record):
        text = super().format(record)
        if not self.styled:
            return text
        color = getattr(record, "color", None)
        if color is None:
            if record.levelno >= logging.ERROR:
                color = "red"
            elif record.levelno >= logging.WARNING:
                color = "yellow"
        return colorize(text, color)


class Glyphs:
    def __init__(self, styled: bool):
        self.styled = styled
        self.rule = "─" if styled else "-"
        self.em = "—" if styled else "-"
        self.dash = "—" if styled else "-"
        self.delta = "ΔE" if styled else "dE"


GLYPHS = Glyphs(False)


def set_style(styled: bool):
    global GLYPHS
    GLYPHS = Glyphs(styled)


def fmt_duration(seconds: float) -> str:
    seconds = max(0.0, seconds)
    if seconds < 60:
        return f"{seconds:.0f}s"
    minutes, secs = divmod(round(seconds), 60)
    if minutes < 60:
        return f"{minutes}m {secs:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes:02d}m"


def fmt_target(value: float) -> str:
    return f"{value:.1e}" if math.isfinite(value) else GLYPHS.dash


def rule(color="dim"):
    return (GLYPHS.rule * WIDTH, color)


def header_block(
    mode, command, fsq_conv, e_conv, max_iter, max_solver_it, saving, started
):
    g = GLYPHS
    title = f" pyRSPthon runs {g.em} SCF driver"
    return [
        rule(),
        (title + started.rjust(WIDTH - len(title)), "bold"),
        rule(),
        (f" {'mode':<13} {mode}", None),
        (f" {'command':<13} {command}", None),
        (
            f" {'fsq target':<13} {fmt_target(fsq_conv):<12} "
            f"{g.delta + ' target':<16} {fmt_target(e_conv)}",
            None,
        ),
        (
            f" {'max iter':<13} {max_iter:<12} {'solver attempts':<16} {max_solver_it}",
            None,
        ),
        (f" {'saving':<13} {saving}", None),
        rule(),
    ]


_ROW = " {it:>4}   {fsq:>10}   {etot:>18}   {delta:>9}   {time:>8}"


def table_header():
    return [
        (
            _ROW.format(
                it="it", fsq="fsq", etot="etot (Ry)", delta=GLYPHS.delta, time="time"
            ),
            "bold",
        ),
    ]


def iteration_row(it, fsq, etot, delta_e, seconds):
    delta = f"{delta_e:.2e}" if math.isfinite(delta_e) else GLYPHS.dash
    return (
        _ROW.format(
            it=it,
            fsq=f"{fsq:.3e}",
            etot=f"{etot:.10f}",
            delta=delta,
            time=fmt_duration(seconds),
        ),
        None,
    )


def stop_line(stop_file):
    mark = "■ " if GLYPHS.styled else ""
    return (f" {mark}Stopped: found abort file {stop_file!r}", "yellow")


def summary_block(
    converged,
    diverged,
    fsq,
    delta_e,
    etot,
    fsq_conv,
    e_conv,
    it,
    performed,
    max_iter,
    elapsed,
):
    g = GLYPHS
    plural = "iteration" if it == 1 else "iterations"
    tail = f" after {it} {plural} ({performed} this run)"
    if diverged:
        word = "✗ Diverged" if g.styled else "DIVERGED"
        head = (f" {word}{tail}", "red")
    elif converged:
        word = "✓ Converged" if g.styled else "CONVERGED"
        head = (f" {word}{tail}", "green")
    else:
        word = "✗ Not converged" if g.styled else "NOT CONVERGED"
        head = (f" {word}{tail} [max iter = {max_iter}]", "yellow")

    fsq_text = f"{fsq:.3e}" if math.isfinite(fsq) else g.dash
    lines = [
        rule(),
        head,
        (f" {'fsq':<6} {fsq_text:<13} (target {fmt_target(fsq_conv)})", None),
    ]
    if math.isfinite(e_conv):
        delta = f"{delta_e:.2e}" if math.isfinite(delta_e) else g.dash
        lines.append(
            (f" {g.delta:<6} {delta:<13} (target {fmt_target(e_conv)})", None)
        )
    if math.isfinite(etot):
        lines.append((f" {'etot':<6} {etot:.10f} Ry", None))
    lines.append((f" total time {fmt_duration(elapsed)}", None))
    lines.append(rule())
    return lines
