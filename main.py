import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import math
import random
import tqdm

def get_complex_at_angle(angle):
    return math.cos(angle) + 1j * math.sin(angle)


tv = get_complex_at_angle(math.pi / 3)

domain = [-4, 4]


# 1 + 2x + x^2 = (x+1)(x+1)
# roots are -1 with multiplicity 2
# degree = 2
# coefficients = np.zeros((degree + 1,), dtype=np.complex64)
# coefficients[0] = 1
# coefficients[1] = tv
# coefficients[2] = tv * tv


def resolve_polynomial(coefficients, roots):
    print(f"coefficients => {coefficients}")
    for r in roots:
        # polyval is a different (old) api and so needs coefficients reversed
        if not (domain[0] <= np.absolute(r) <= domain[1]):
            # outside solution domain, but would be better to just not compute them
            continue
        result = np.polyval(coefficients[::-1], r)
        print(f"{r} => {result}")

all_roots = [] 

# overwrite coefficients with whatever

# polynomial with degree 11
degree = 11
coefficients = np.zeros((degree + 1,), dtype=np.complex64)

# constant coefficients
coefficients[11] = 1
coefficients[10] = -1

N = 10000
n_frames = 24*5

t1s = np.array([get_complex_at_angle(2*math.pi*random.random()) for _ in range(N)])
t2s = np.array([get_complex_at_angle(2*math.pi*random.random()) for _ in range(N)])

coeffs8 = 30j * t1s * t1s - 30 * t1s - 30
coeffs5 = 30j * t2s * t2s + 30j * t2s - 30

frames = []
for fi in tqdm.tqdm(range(n_frames)):
    frames.append([])
    c_frame = frames[-1]

    tv = get_complex_at_angle(2 * math.pi * (fi / n_frames))
    coefficients[6] = -30 * (tv**5) - 30j * (tv**3) + 30j * (tv * tv) - 30j * tv + 30

    for i in range(N):
        coefficients[8] = coeffs8[i]
        coefficients[5] = coeffs5[i]

        roots = np.polynomial.polynomial.polyroots(coefficients)
        roots = roots.astype(np.complex64)
        for r in roots:
            if np.isclose(r, 0+0j, atol=1e-4, rtol=1e-4):
                continue
            #if not (domain[0] <= np.absolute(r) <= domain[1]):
            #    continue
            #if not np.isclose(np.polyval(coefficients[::-1], r), 0, atol=1e-4, rtol=1e-4):
            #    continue
            
            c_frame.append(r)

        frames[-1] = np.asarray(c_frame, dtype=np.complex64)

assert len(frames) == n_frames

all_frames = np.concatenate(frames, axis=0)
x_min, x_max = np.min(all_frames.real), np.max(all_frames.real)
y_min, y_max = np.min(all_frames.imag), np.max(all_frames.imag)

#x_min *= 0.8
#x_max *= 0.8
#y_min *= 0.8
#y_max *= 0.8

hists = []
for frame in frames:
    hist, *_ = np.histogram2d(frame.real, frame.imag, bins=200, range=[[x_min, x_max], [y_min, y_max]])
    hists.append(hist)

fig, ax = plt.subplots(figsize = (8,8))
fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)
fig.set_size_inches(4, 4, True)
ax.set_axis_off()
im = plt.imshow(hists[0], cmap='gray', origin='lower')

def animate(i):
    im.set_data(hists[i])

anim = animation.FuncAnimation(fig, animate, frames=len(hists), interval=int(1000 / 24), blit=False)
#FFwriter = animation.FFMpegWriter(fps=24, bitrate=8_000_000)
#anim.save('animation.mp4', writer=FFwriter, dpi=800)
anim.save('animation.gif', dpi=300)

