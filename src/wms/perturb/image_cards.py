"""ImageNet-C severity constants; NumPy-only RGB transforms for S1 cards.

Brightness adds to HSV value (preserving hue/saturation). Contrast is about
per-channel spatial means. Color bias is our own red/blue shift, not ImageNet-C.
Existing Perturbations operations are not changed.
"""
import numpy as np


def severity_index(severity):
    if type(severity) is not int or not 0 <= severity <= 5:
        raise ValueError('severity must be an integer in 0..5')
    return severity - 1


def _uint8(value):
    return np.clip(np.rint(value * 255), 0, 255).astype(np.uint8)


def brightness(image, severity):
    index = severity_index(severity)
    if severity == 0:
        return image
    x = image.astype(float) / 255
    value = x.max(axis=-1, keepdims=True)
    raised = np.minimum(1, value + (.1, .2, .3, .4, .5)[index])
    return _uint8(np.where(value > 0, x * raised / np.maximum(value, 1e-12), raised))


def contrast(image, severity):
    index = severity_index(severity)
    if severity == 0:
        return image
    x = image.astype(float) / 255
    mean = x.mean(axis=(0, 1), keepdims=True)
    return _uint8((x - mean) * (.4, .3, .2, .1, .05)[index] + mean)


def color_bias(image, severity):
    index = severity_index(severity)
    if severity == 0:
        return image
    shift = (.04, .08, .12, .18, .25)[index]
    return _uint8(image.astype(float) / 255 + np.array([shift, 0, -shift]))


def affine(image, dx=0, dy=0, angle=0, scale=1):
    """Inverse mapping, bilinear interpolation, edge padding; dx/dy in pixels."""
    values = np.array([dx, dy, angle, scale], dtype=float)
    if not np.isfinite(values).all() or scale <= 0:
        raise ValueError('affine arguments must be finite and scale positive')
    if dx == dy == angle == 0 and scale == 1:
        return image
    h, w = image.shape[:2]
    y, x = np.indices((h, w), dtype=float)
    x, y = x - (w-1)/2 - dx, y - (h-1)/2 - dy
    c, s = np.cos(np.deg2rad(angle)), np.sin(np.deg2rad(angle))
    u = np.clip((c*x+s*y)/scale+(w-1)/2, 0, w-1)
    v = np.clip((-s*x+c*y)/scale+(h-1)/2, 0, h-1)
    x0, y0 = np.floor(u).astype(int), np.floor(v).astype(int)
    x1, y1 = np.minimum(x0+1, w-1), np.minimum(y0+1, h-1)
    a, b = (u-x0)[..., None], (v-y0)[..., None]
    out = ((1-a)*(1-b)*image[y0,x0] + a*(1-b)*image[y0,x1]
           + (1-a)*b*image[y1,x0] + a*b*image[y1,x1])
    return np.clip(np.rint(out), 0, 255).astype(np.uint8)
