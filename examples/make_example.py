#!/usr/bin/env python3
"""Create three synthetic 40-frame shots using only Python's standard library."""
import argparse
import json
import math
import struct
import zlib
from pathlib import Path


def png(width, height, boxes):
    """Draw colored ellipses in a tiny RGB PNG; this is sample data, not footage."""
    pixels = bytearray(bytes((12, 30, 55)) * width * height)
    for (x, y, w, h), color in boxes:
        for yy in range(y, y+h):
            for xx in range(x, x+w):
                if ((xx + .5 - x - w/2)/(w/2))**2 + ((yy + .5 - y - h/2)/(h/2))**2 <= 1:
                    offset = (yy*width+xx)*3
                    pixels[offset:offset+3] = bytes(color)
    raw = b"".join(b"\0" + pixels[y*width*3:(y+1)*width*3] for y in range(height))

    def chunk(kind, contents):
        return struct.pack(">I", len(contents)) + kind + contents + struct.pack(">I", zlib.crc32(kind+contents))

    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))


def make_example(output):
    output = Path(output)
    if output.exists():
        raise ValueError(f"Output exists; choose a new directory: {output}")
    output.mkdir(parents=True)
    images, annotations = [], []
    for shot in ("demo_a", "demo_b", "demo_c"):
        for number in range(40):
            name = f"{shot}-{number:04d}.png"
            boxes = [(20+number*2, 40+round(8*math.sin(number/8)), 18, 10),
                     (180-number, 100+round(6*math.cos(number/9)), 22, 12)]
            (output / name).write_bytes(png(256, 160, list(zip(boxes, ((255, 190, 40), (70, 220, 180))))))
            image_id = len(images)
            images.append(dict(id=image_id, file_name=name, width=256, height=160))
            for category, box in enumerate(boxes, 1):
                annotations.append(dict(id=len(annotations), image_id=image_id, category_id=category, bbox=list(box)))
    # Reverse image entries to exercise chronological sorting of COCO exports.
    data = dict(images=list(reversed(images)), annotations=annotations,
                categories=[dict(id=1, name="100"), dict(id=2, name="101")])
    (output / "_annotations.coco.json").write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/example_source"))
    args = parser.parse_args()
    try:
        print(f"Created 120 synthetic frames in {make_example(args.output)}")
    except (OSError, ValueError) as error:
        parser.exit(1, f"ERROR: {error}\n")
