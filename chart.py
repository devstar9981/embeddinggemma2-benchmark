"""Draws results/ndcg-vs-dims.png from results/quality-eg1.json and results/quality-eg2.json.

    python chart.py

Two small multiples (CodeSearchNet averaged over 6 languages, and CosQA), NDCG@10 at each vector
size for both models. Rendered as SVG in a page and screenshotted with headless Chrome or Edge.
"""

import json
import os
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).parent
RESULTS = ROOT / "results"
DIMS = ["768", "512", "256", "128"]
CSN = ["csn-python", "csn-javascript", "csn-go", "csn-ruby", "csn-java", "csn-php"]

# Validated categorical slots 1 and 2 (light surface), plus chart ink.
SURFACE, INK, INK_2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SERIES = [
    ("eg2", "EmbeddingGemma 2 (text only, 271M)", "#2a78d6"),
    ("eg1", "EmbeddingGemma (308M)", "#eb6834"),
]
W, H = 1200, 700
FONT = 'system-ui, -apple-system, "Segoe UI", sans-serif'


def scores(data, tasks):
    return [sum(data["tasks"][t][d]["ndcg@10"] for t in tasks) / len(tasks) for d in DIMS]


def nice_range(values, step):
    lo = (min(values) // step) * step
    hi = -(-max(values) // step) * step
    return lo - (step if max(values) - lo < step else 0), hi + (step if hi - max(values) < step / 2 else 0)


def panel(x0, y0, w, h, title, note, lines, step):
    """One small multiple: lines is [(label, color, [score per dim])]."""
    lo, hi = nice_range([v for _, _, vals in lines for v in vals], step)
    left, right, top, bottom = 56, 110, 70, 46
    inset = 64  # room between the y-axis numbers and the first point, for its value label
    pw, ph = w - left - right, h - top - bottom
    px = lambda i: x0 + left + inset + (pw - inset) * i / (len(DIMS) - 1)  # noqa: E731
    py = lambda v: y0 + top + ph * (hi - v) / (hi - lo)  # noqa: E731
    out = [
        f'<text x="{x0}" y="{y0 + 22}" font-size="22" font-weight="600" fill="{INK}">{title}</text>',
        f'<text x="{x0}" y="{y0 + 48}" font-size="17" fill="{INK_2}">{note}</text>',
    ]
    tick = lo
    while tick <= hi + 1e-9:
        y = py(tick)
        out.append(f'<line x1="{x0 + left}" x2="{x0 + left + pw}" y1="{y:.1f}" y2="{y:.1f}" stroke="{GRID}" stroke-width="1"/>')
        out.append(
            f'<text x="{x0 + left - 12}" y="{y + 6:.1f}" font-size="16" fill="{MUTED}" text-anchor="end" '
            f'style="font-variant-numeric: tabular-nums">{tick:g}</text>'
        )
        tick += step
    out.append(f'<line x1="{x0 + left}" x2="{x0 + left + pw}" y1="{py(lo):.1f}" y2="{py(lo):.1f}" stroke="{AXIS}" stroke-width="1"/>')
    for i, d in enumerate(DIMS):
        out.append(f'<text x="{px(i):.1f}" y="{y0 + h - 14}" font-size="16" fill="{MUTED}" text-anchor="middle">{d}</text>')
    for _, color, vals in lines:
        path = " ".join(f"{'M' if i == 0 else 'L'}{px(i):.1f},{py(v):.1f}" for i, v in enumerate(vals))
        out.append(f'<path d="{path}" fill="none" stroke="{color}" stroke-width="2" stroke-linejoin="round" stroke-linecap="round"/>')
        for i, v in enumerate(vals):
            out.append(f'<circle cx="{px(i):.1f}" cy="{py(v):.1f}" r="5" fill="{color}" stroke="{SURFACE}" stroke-width="2"/>')
    # Direct labels at both ends; nudge apart when the two series end close together.
    for end, anchor, dx in ((0, "end", -12), (len(DIMS) - 1, "start", 12)):
        ys = sorted(((py(vals[end]), vals[end]) for _, _, vals in lines))
        if len(ys) == 2 and ys[1][0] - ys[0][0] < 22:
            mid = (ys[0][0] + ys[1][0]) / 2
            ys = [(mid - 11, ys[0][1]), (mid + 11, ys[1][1])]
        for y, v in ys:
            out.append(
                f'<text x="{px(end) + dx:.1f}" y="{y + 6:.1f}" font-size="17" font-weight="600" fill="{INK}" '
                f'text-anchor="{anchor}">{v:.1f}</text>'
            )
    return out


def main():
    eg = {key: json.loads((RESULTS / f"quality-{key}.json").read_text()) for key, _, _ in SERIES}
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" font-family=\'{FONT}\'>',
        f'<rect width="{W}" height="{H}" fill="{SURFACE}"/>',
        f'<text x="48" y="64" font-size="30" font-weight="600" fill="{INK}">Code search quality as the vectors get shorter</text>',
        f'<text x="48" y="98" font-size="19" fill="{INK_2}">NDCG@10, higher is better. Same queries and code for both models, float32.</text>',
    ]
    lx = 48
    for _, label, color in SERIES:
        svg.append(f'<circle cx="{lx + 7}" cy="134" r="6" fill="{color}"/>')
        svg.append(f'<text x="{lx + 22}" y="140" font-size="18" fill="{INK}">{label}</text>')
        lx += 22 + len(label) * 9.6 + 40
    svg += panel(
        48, 170, 540, 470, "CodeSearchNet", "6,000 docstring queries, average of 6 languages",
        [(label, color, scores(eg[key], CSN)) for key, label, color in SERIES], 2,
    )
    svg += panel(
        620, 170, 540, 470, "CosQA", "500 real web queries over 20,604 functions",
        [(label, color, scores(eg[key], ["cosqa"])) for key, label, color in SERIES], 2,
    )
    svg.append(f'<text x="{W / 2}" y="{H - 14}" font-size="16" fill="{MUTED}" text-anchor="middle">Vector dimensions</text>')
    svg.append("</svg>")

    html = f'<!doctype html><meta charset="utf-8"><style>html,body{{margin:0;background:{SURFACE}}}</style>{"".join(svg)}'
    (RESULTS / "ndcg-vs-dims.svg").write_text("".join(svg), encoding="utf8")
    browser = next(
        p
        for p in [
            os.environ.get("CHROME_PATH", ""),
            r"C:\Program Files\Google\Chrome\Application\chrome.exe",
            r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
            "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
            "/usr/bin/google-chrome",
            "/usr/bin/chromium",
        ]
        if p and Path(p).exists()
    )
    with tempfile.TemporaryDirectory() as tmp:
        page = Path(tmp) / "chart.html"
        page.write_text(html, encoding="utf8")
        subprocess.run(
            [
                browser,
                "--headless=new",
                "--hide-scrollbars",
                "--force-device-scale-factor=2",
                f"--window-size={W},{H}",
                f"--user-data-dir={Path(tmp) / 'profile'}",
                f"--screenshot={RESULTS / 'ndcg-vs-dims.png'}",
                page.as_uri(),
            ],
            check=True,
            capture_output=True,
            timeout=120,
        )
    print("wrote results/ndcg-vs-dims.png and .svg")


if __name__ == "__main__":
    main()
