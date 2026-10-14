Stable fish dataset, sampled at 30 FPS.
Splits are fixed in manifest.json. Whole chunks are assigned in chronological order
within each source; supplied source-folder order is preserved. Same-recording chunks
may be in different splits. Requested ratios are approximate.
A chunk requires consecutive frame numbers and identical annotated fish IDs throughout.
Any missing/added fish box or ambiguous identity breaks the chunk.
Annotations: image, class_id=0 (fish), track_id, bbox, center. Coordinates are pixels.
Track identity is scoped to the source recording; IDs can repeat in other recordings.
Centers are bounding-box midpoints. Source files are preserved.
Only stable chunks meeting the minimum length are included in this package.
Use prediction_loader.py within one split; it never crosses chunk boundaries.
