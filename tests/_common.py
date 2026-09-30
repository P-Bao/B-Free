import os, sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CODE = os.path.join(ROOT, "code")
if CODE not in sys.path:
    sys.path.insert(0, CODE)
# 112 px -> 8x8 patch grid: GSR window=3 masks a 4x4 grid entirely (softmax NaN); real 504 -> 36x36.
IMG = 112
