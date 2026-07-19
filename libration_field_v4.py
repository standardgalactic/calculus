import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
from scipy.interpolate import LinearNDInterpolator
from scipy.ndimage import map_coordinates
from PIL import Image

deg = np.pi / 180.0

# ------------------------------------------------------------------
# 1. Simplified analytic optical-libration model (swap for a real
#    DE440/Horizons ephemeris for publication-grade precision — no
#    network access to Horizons/SPICE from this sandbox)
# ------------------------------------------------------------------
def mean_elements(t_days):
    Mp = (134.9634 + 13.0649929509 * t_days) * deg
    F  = (93.2721  + 13.2293504490 * t_days) * deg
    D  = (297.8502 + 12.1907491914 * t_days) * deg
    M  = (357.5291 + 0.9856003     * t_days) * deg
    return Mp, F, D, M

def sub_earth_point(t_days):
    Mp, F, D, M = mean_elements(t_days)
    l = (-6.29 * np.sin(Mp) + 1.27 * np.sin(2 * D - Mp)
         - 0.23 * np.sin(2 * D + Mp) - 0.17 * np.sin(2 * D))
    b = 6.85 * np.sin(F) - 0.17 * np.sin(F - 2 * D)
    return l * deg, b * deg

t = np.arange(0, 19 * 365.25, 0.5)
l_t, b_t = sub_earth_point(t)
ex, ey, ez = np.cos(b_t) * np.cos(l_t), np.cos(b_t) * np.sin(l_t), np.sin(b_t)
E = np.stack([ex, ey, ez], axis=0)
T = E.shape[1]

# ------------------------------------------------------------------
# 2. Surface grid
# ------------------------------------------------------------------
lat = np.arange(-90, 90.01, 1.0)
lon = np.arange(-180, 180.01, 1.0)
LON, LAT = np.meshgrid(lon, lat)
phi = LAT.ravel() * deg
lam = LON.ravel() * deg
P = np.stack([np.cos(phi) * np.cos(lam), np.cos(phi) * np.sin(lam), np.sin(phi)], axis=1)
N = P.shape[0]

count = np.zeros(N)
maxdot = np.full(N, -2.0)
sum_dotpos = np.zeros(N)
sum_l = np.zeros(N)
sum_b = np.zeros(N)

chunk = 1000
for i in range(0, T, chunk):
    Ec = E[:, i:i+chunk]
    lc, bc = l_t[i:i+chunk], b_t[i:i+chunk]
    dots = P @ Ec
    vis = dots > 0
    count += vis.sum(axis=1)
    maxdot = np.maximum(maxdot, dots.max(axis=1))
    sum_dotpos += np.where(vis, dots, 0).sum(axis=1)
    sum_l += (vis * lc).sum(axis=1)
    sum_b += (vis * bc).sum(axis=1)

Fvis = (count / T).reshape(LAT.shape)
Mmargin = maxdot.reshape(LAT.shape)
Vvis = Mmargin > 0
mean_dot = (sum_dotpos / T).reshape(LAT.shape)
count_grid = count.reshape(LAT.shape)
with np.errstate(invalid='ignore', divide='ignore'):
    mean_l = np.where(count_grid > 0, (sum_l / np.maximum(count, 1)).reshape(LAT.shape), 0)
    mean_b = np.where(count_grid > 0, (sum_b / np.maximum(count, 1)).reshape(LAT.shape), 0)

w = np.cos(LAT * deg)
frac_area = (Vvis * w).sum() / w.sum()
print(f"Area-weighted visible fraction ~ {frac_area*100:.1f}%")

# ------------------------------------------------------------------
# 3. Equal-area azimuthal projection
# ------------------------------------------------------------------
phi_r, lam_r = LAT * deg, LON * deg
cosc = np.clip(np.cos(phi_r) * np.cos(lam_r), -1, 1)
c = np.arccos(cosc)
theta = np.arctan2(np.sin(phi_r), np.cos(phi_r) * np.sin(lam_r))
c_max = c[Vvis].max()
r = np.sqrt(np.clip((1 - np.cos(c)) / (1 - np.cos(c_max)), 0, None))
X, Y = r * np.cos(theta), r * np.sin(theta)

# ------------------------------------------------------------------
# 4. REAL lunar albedo texture, sampled at the same lat/lon grid.
#
# Source: mrdoob/three.js examples/textures/planets/moon_1024.jpg —
# an equirectangular nearside-centered lunar mosaic derived from the
# Clementine spacecraft dataset (USGS Map-a-Planet, public domain).
# Downloaded once via raw.githubusercontent.com (see fetch_texture.sh
# alongside this script) and cached locally as moon_1024.jpg.
#
# Equirectangular convention: column 0..W-1 spans longitude -180..+180
# (prime meridian, lon=0, at the centre column, matching our sub-Earth
# convention); row 0..H-1 spans latitude +90..-90.
# ------------------------------------------------------------------
tex_img = np.asarray(Image.open('moon_1024.jpg').convert('L'), dtype=np.float64) / 255.0
Himg, Wimg = tex_img.shape

col = (LON + 180.0) / 360.0 * (Wimg - 1)
row = (90.0 - LAT) / 180.0 * (Himg - 1)
albedo = map_coordinates(tex_img, [row, col], order=1, mode='wrap')

# mild contrast stretch so mid-tone maria/highland contrast reads clearly
alo, ahi = np.percentile(albedo, [1, 99])
albedo = np.clip((albedo - alo) / (ahi - alo), 0, 1)
albedo = 0.35 + 0.65 * albedo  # keep it a modulation, not full black

# ------------------------------------------------------------------
# 5. Colour: hue = direction of libration revealing the terrain,
#    saturation = how marginal it is, value = expected view quality
#    modulated by the real albedo texture
# ------------------------------------------------------------------
angle = np.arctan2(mean_b, mean_l)
hue = (angle / (2 * np.pi)) % 1.0
sat = np.clip((1 - Fvis), 0, 1) ** 0.7

val = np.log1p(15 * np.clip(mean_dot, 0, None)) / np.log(16)
val = np.clip(val, 0, 1) * albedo
val = np.clip(val, 0, 1)

hsv = np.stack([hue, sat, val], axis=-1)
rgb = hsv_to_rgb(hsv)
rgb[~Vvis] = 0

# ------------------------------------------------------------------
# 6. Smooth continuous raster via a single vector-valued triangulation
# ------------------------------------------------------------------
mask = Vvis
pts = np.column_stack([X[mask], Y[mask]])
vals = rgb[mask]
interp = LinearNDInterpolator(pts, vals, fill_value=0.0)

res = 1000
xi = np.linspace(-1.05, 1.05, res)
yi = np.linspace(-1.05, 1.05, res)
XI, YI = np.meshgrid(xi, yi)
img = interp(XI, YI)
img = np.nan_to_num(img, nan=0.0)

dist = np.sqrt(XI**2 + YI**2)
img[dist > 1.0] = 0.0

# ------------------------------------------------------------------
# 7. Plot
# ------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 10), facecolor='black')
ax.set_facecolor('black')
ax.imshow(img, extent=[-1.05, 1.05, -1.05, 1.05], origin='lower', interpolation='bilinear')

circle = plt.Circle((0, 0), 1.0, fill=False, edgecolor='#555555', linewidth=0.8, alpha=0.7)
ax.add_patch(circle)

ax.set_xlim(-1.05, 1.05)
ax.set_ylim(-1.05, 1.05)
ax.set_aspect('equal')
ax.axis('off')
ax.set_title("The Fifty-Ninth Percent \u2014 Earth's Accumulated View of the Moon",
              color='#dddddd', fontsize=13, pad=14)
ax.text(0, -1.14,
        f'area-weighted visible fraction \u2248 {frac_area*100:.1f}%   \u00b7   '
        'albedo: Clementine mosaic (USGS/three.js, public domain)\n'
        'hue = direction of libration revealing the terrain (E\u2192red, N\u2192green, W\u2192blue, S\u2192violet)   \u00b7   '
        'saturation = marginality   \u00b7   brightness = expected view quality \u00d7 albedo',
        color='#888888', fontsize=8, ha='center')

plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/fifty_ninth_percent_v4.png', dpi=200,
            facecolor='black', bbox_inches='tight')
print("saved")
