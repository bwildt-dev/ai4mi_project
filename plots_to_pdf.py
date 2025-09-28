import argparse
from pathlib import Path
from math import ceil, sqrt
from PIL import Image
from reportlab.lib.pagesizes import A4, landscape
from reportlab.pdfgen import canvas


def get_png_files(directory: Path):
    return sorted(directory.glob('*.png'))


def arrange_grid(n):
    cols = ceil(sqrt(n))
    rows = ceil(n / cols)
    return rows, cols


def add_images_to_pdf(png_files, pdf_path):
    page_width, page_height = landscape(A4)
    n = len(png_files)
    if n == 0:
        print("No PNG files found.")
        return
    rows, cols = arrange_grid(n)
    c = canvas.Canvas(str(pdf_path), pagesize=landscape(A4))
    margin = 20
    cell_width = (page_width - 2 * margin) / cols
    cell_height = (page_height - 2 * margin) / rows
    for idx, img_path in enumerate(png_files):
        row = idx // cols
        col = idx % cols
        x = margin + col * cell_width
        y = page_height - margin - (row + 1) * cell_height
        with Image.open(img_path) as img:
            img_w, img_h = img.size
            scale = min(cell_width / img_w, cell_height / img_h)
            new_w = int(img_w * scale)
            new_h = int(img_h * scale)
            img = img.resize((new_w, new_h), Image.LANCZOS)
            temp_path = img_path.parent / (".temp_" + img_path.name)
            img.save(temp_path)
            c.drawImage(str(temp_path), x + (cell_width - new_w) / 2, y + (cell_height - new_h) / 2, width=new_w, height=new_h)
            temp_path.unlink()
    c.showPage()
    c.save()
    print(f"Saved PDF to {pdf_path}")


def main():
    parser = argparse.ArgumentParser(description="Combine all PNGs in a directory onto a single A4 PDF page.")
    parser.add_argument("dir", type=Path, help="Directory containing PNG files.")
    args = parser.parse_args()
    png_files = get_png_files(args.dir)
    if not png_files:
        print("No PNG files found in the directory.")
        return
    pdf_path = args.dir / "output.pdf"
    add_images_to_pdf(png_files, pdf_path)


if __name__ == "__main__":
    main()
