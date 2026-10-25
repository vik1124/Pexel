from PIL import Image, ImageDraw, ImageFont
from pathlib import Path

CHARTS_DIR = Path('output/charts/high')
OUT = Path('output/charts/collage_high.png')

COLS = 6
THUMB = (400, 300)       # each thumbnail
LABEL_H = 28             # height reserved for label below each thumb
PAD = 8                  # padding between cells
BG = (245, 245, 248)
LABEL_BG = (30, 30, 40)
LABEL_FG = (220, 220, 230)

pngs = sorted(CHARTS_DIR.glob('*.png'))
rows = -(-len(pngs) // COLS)   # ceil division

cell_w = THUMB[0] + PAD
cell_h = THUMB[1] + LABEL_H + PAD
canvas_w = COLS * cell_w + PAD
canvas_h = rows * cell_h + PAD

canvas = Image.new('RGB', (canvas_w, canvas_h), BG)
draw = ImageDraw.Draw(canvas)

try:
    font = ImageFont.truetype('C:/Windows/Fonts/consola.ttf', 15)
except Exception:
    font = ImageFont.load_default()

for idx, path in enumerate(pngs):
    col = idx % COLS
    row = idx // COLS
    x = PAD + col * cell_w
    y = PAD + row * cell_h

    img = Image.open(path).convert('RGB')
    img.thumbnail(THUMB, Image.LANCZOS)

    # centre thumbnail in cell
    ox = x + (THUMB[0] - img.width) // 2
    oy = y + (THUMB[1] - img.height) // 2
    canvas.paste(img, (ox, oy))

    # label strip
    label_y = y + THUMB[1]
    draw.rectangle([x, label_y, x + THUMB[0], label_y + LABEL_H], fill=LABEL_BG)
    name = path.stem.replace('_', ' ')
    bbox = draw.textbbox((0, 0), name, font=font)
    tw = bbox[2] - bbox[0]
    tx = x + (THUMB[0] - tw) // 2
    ty = label_y + (LABEL_H - (bbox[3] - bbox[1])) // 2
    draw.text((tx, ty), name, fill=LABEL_FG, font=font)

canvas.save(OUT, format='PNG', optimize=True)
print(f'Saved {OUT}  ({canvas_w}x{canvas_h}px,  {len(pngs)} charts)')
