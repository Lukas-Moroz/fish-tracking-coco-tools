#!/usr/bin/env python3
"""Audit COCO fish boxes and consecutive identity trajectories without editing data."""
import argparse
import csv
import json
import math
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image


def center(a):
    x, y, w, h = a['bbox']
    return x + w / 2, y + h / 2


def distance(a, b):
    return math.dist(center(a), center(b))


def audit(source, output):
    output.mkdir(parents=True, exist_ok=True)
    data = json.loads((source / '_annotations.coco.json').read_text())
    categories = {c['id']: c['name'] for c in data['categories']}
    images = {im['id']: im for im in data['images']}
    by_frame = defaultdict(lambda: defaultdict(list))
    issues, transitions, accelerations = [], [], []
    for section in ('images', 'annotations', 'categories'):
        duplicates = [key for key, count in Counter(x['id'] for x in data[section]).items() if count > 1]
        if duplicates:
            issues.append(f'Duplicate {section} IDs: {duplicates}')
    frames = {}
    for im in data['images']:
        name = (im.get('extra') or {}).get('name', im['file_name'])
        number = int(re.search(r'-(\d+)', name).group(1))
        if number in frames:
            issues.append(f'Duplicate frame number: {number}')
        frames[number] = im
        with Image.open(source / im['file_name']) as photo:
            if photo.size != (im['width'], im['height']):
                issues.append(f'Frame {number}: image dimension mismatch {photo.size}')
            photo.verify()
    for a in data['annotations']:
        if a['image_id'] not in images or a['category_id'] not in categories:
            issues.append(f'Annotation {a["id"]}: invalid reference')
            continue
        im = images[a['image_id']]
        number = int(re.search(r'-(\d+)', (im.get('extra') or {}).get('name', im['file_name'])).group(1))
        bbox = a.get('bbox')
        if not isinstance(bbox, list) or len(bbox) != 4 or not all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in bbox):
            issues.append(f'Annotation {a["id"]}: invalid bbox')
            continue
        x, y, w, h = bbox
        if w <= 0 or h <= 0 or x < 0 or y < 0 or x + w > im['width'] or y + h > im['height']:
            issues.append(f'Frame {number}, fish {categories[a["category_id"]]}, annotation {a["id"]}: out-of-bounds/nonpositive box {bbox}')
        by_frame[number][a['category_id']].append(a)
    numbers = sorted(frames)
    if not numbers:
        raise ValueError("No source frames to audit.")
    missing = sorted(set(range(numbers[0], numbers[-1] + 1)) - set(numbers))
    for number, fish in sorted(by_frame.items()):
        for cid, anns in fish.items():
            if len(anns) > 1:
                issues.append(f'Frame {number}, fish {categories[cid]}: duplicate identity, annotation IDs {[a["id"] for a in anns]}, boxes {[a["bbox"] for a in anns]}')
    for number in numbers:
        previous = by_frame.get(number - 1, {})
        current = by_frame[number]
        for cid in sorted(set(previous) & set(current)):
            if len(previous[cid]) != 1 or len(current[cid]) != 1:
                continue
            a, b = previous[cid][0], current[cid][0]
            if any(box['bbox'][2] <= 0 or box['bbox'][3] <= 0 for box in (a, b)):
                continue
            step = distance(a, b)
            area_a, area_b = a['bbox'][2] * a['bbox'][3], b['bbox'][2] * b['bbox'][3]
            ratio = max(area_a, area_b) / min(area_a, area_b)
            norm = step / math.hypot(a['bbox'][2], a['bbox'][3])
            transitions.append(dict(fish=categories[cid], start=number - 1, end=number,
                                    displacement_px=step, displacement_box_diagonals=norm,
                                    area_ratio=ratio, previous_bbox=a['bbox'], current_bbox=b['bbox']))
            before = by_frame.get(number - 2, {}).get(cid, [])
            if len(before) == 1 and before[0]['bbox'][2] > 0 and before[0]['bbox'][3] > 0:
                c0, c1, c2 = center(before[0]), center(a), center(b)
                acceleration = math.hypot(c2[0] - 2*c1[0] + c0[0], c2[1] - 2*c1[1] + c0[1])
                accelerations.append(dict(fish=categories[cid], frame=number - 1, acceleration_px=acceleration))
    with (output / 'transitions.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["fish", "start", "end", "displacement_px",
                                              "displacement_box_diagonals", "area_ratio",
                                              "previous_bbox", "current_bbox"])
        writer.writeheader()
        writer.writerows(transitions)
    report = [f"Images: {len(data['images'])}; annotations: {len(data['annotations'])}",
              f'Frame range: {numbers[0]}-{numbers[-1]}; missing frames: {missing}',
              '', 'Structural issues:', *issues, '', 'Consecutive displacement statistics by fish:']
    for cid, label in sorted(categories.items()):
        rows = [t for t in transitions if t['fish'] == label]
        if not rows:
            continue
        steps = sorted(t['displacement_px'] for t in rows)
        report.append(f"Fish {label}: n={len(rows)}, median={statistics.median(steps):.2f}px, "
                      f"p95={steps[math.ceil(.95*len(steps))-1]:.2f}px, max={max(steps):.2f}px")
    for title, key in [('Largest movements', 'displacement_px'), ('Largest movements relative to box diagonal', 'displacement_box_diagonals'), ('Largest box area changes', 'area_ratio')]:
        report.extend(['', title + ':'])
        for row in sorted(transitions, key=lambda t: t[key], reverse=True)[:20]:
            report.append(f"  fish {row['fish']} {row['start']:04d}->{row['end']:04d}: "
                          f"{row['displacement_px']:.2f}px, {row['displacement_box_diagonals']:.2f} diagonals, "
                          f"area ratio {row['area_ratio']:.2f}; {row['previous_bbox']} -> {row['current_bbox']}")
    report.extend(['', 'Largest changes in velocity (pixels per frame squared):'])
    report.extend(str(row) for row in sorted(accelerations, key=lambda a: a['acceleration_px'], reverse=True)[:20])
    report.extend(['', 'Presence runs and start/end edge distances:'])
    for cid, label in sorted(categories.items()):
        runs = []
        for n in numbers:
            if cid not in by_frame[n]:
                continue
            if runs and runs[-1][-1] == n - 1:
                runs[-1].append(n)
            else:
                runs.append([n])
        for run in runs:
            edges = []
            for n in (run[0], run[-1]):
                x, y, w, h = by_frame[n][cid][0]['bbox']
                im = frames[n]
                edges.append(min(x, y, im['width'] - x - w, im['height'] - y - h))
            report.append(f'Fish {label}: {run[0]:04d}-{run[-1]:04d}, start/end box distance to closest image edge: {edges}')
    (output / 'numeric_audit.txt').write_text('\n'.join(report) + '\n')
    print('\n'.join(report))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    try:
        audit(args.source, args.output)
    except (OSError, ValueError, KeyError) as error:
        parser.exit(1, f"ERROR: {error}\n")
