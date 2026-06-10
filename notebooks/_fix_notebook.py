import json, re

with open('notebooks/05_bbox_difficulty_analysis.ipynb', 'rb') as f:
    raw = f.read()

# Replace non-ASCII chars with their unicode escape sequences
decoded = raw.decode('latin-1')

# Find what's at position 27959 (the error location)
print(f"Character at 27959: {repr(decoded[27959])}")
print(f"Context: {repr(decoded[27950:27970])}")
