import math

import cv2
import matplotlib.pyplot as plt
import numpy as np
import time
import os

os.makedirs('images', exist_ok=True)
os.makedirs('edges', exist_ok=True)

def normalize_edge_map(edge_map):
    edge_map = np.abs(edge_map)
    return cv2.normalize(edge_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

def log_kernel(ksize=7, sigma=1.0):
    assert ksize % 2 == 1
    half = ksize // 2
    x = np.arange(-half, half + 1)
    y = np.arange(-half, half + 1)
    X, Y = np.meshgrid(x, y)
    R2 = X**2 + Y**2
    normalization = (R2 - 2 * sigma**2) / (sigma**4)
    gaussian = np.exp(-R2 / (2 * sigma**2))
    kernel = normalization * gaussian
    kernel -= kernel.mean()  # zero-center
    return kernel.astype(np.float32)

def gaussian_kernel(ksize=7, sigma=1.0):
    """Generate a 2D Gaussian kernel."""
    assert ksize % 2 == 1
    half = ksize // 2
    x = np.arange(-half, half + 1)
    y = np.arange(-half, half + 1)
    X, Y = np.meshgrid(x, y)
    kernel = np.exp(-(X**2 + Y**2) / (2 * sigma**2))
    kernel /= kernel.sum()
    return kernel

def q_gaussian_kernel(ksize = 7, sigma = 1.0, q = 1.0):
    """
    Generate a 2D q-Gaussian kernel.

    Args:
        ksize (int): kernel size (odd number, e.g. 3, 5, 7).
        sigma (float): scale parameter (the sigma for Gaussian).
        q (float): entropic index (q=1 → normal Gaussian).

    Returns:
        kernel (numpy.ndarray): normalized q-Gaussian kernel.
    """
    assert ksize % 2 == 1
    half = ksize // 2
    x = np.arange(-half, half + 1)
    X, Y = np.meshgrid(x, x)
    R2 = X**2 + Y**2

    """
    if q<1:
        Cq = (2 * math.sqrt(math.pi) * math.gamma((1) / (1 - q))) / (
                    (3 - q) * math.sqrt(1 - q) * math.gamma((3 - q) / (2 * (1 - q))))
    elif q>1:
        Cq = (math.sqrt(math.pi) * math.gamma((3 - q) / (2 * (q - 1))) / (
                    (3 - q) * math.sqrt(1 - q) * math.gamma((1) / (q - 1))))
    else:
        Cq = math.sqrt(math.pi)
        
    expq = 

    """
    beta = 1.0 / (math.sqrt(2 * sigma**2))
    if np.isclose(q, 1.0):
        kernel = np.exp(-beta * R2)
    else:
        base = 1 - (1 - q) * beta * R2
        base = np.maximum(base, 0.0)
        kernel = base ** (1.0 / (1.0 - q))
    s = kernel.sum()
    if s == 0:
        return kernel.astype(np.float32)
    return (kernel / s).astype(np.float32)


def Sobel(img_blur, ksize):
    # -------------------------------------
    print('Sobel')
    # -------------------------------------

    start = time.time()

    gx = cv2.Sobel(img_blur, cv2.CV_64F, 1, 0, ksize=ksize)
    gy = cv2.Sobel(img_blur, cv2.CV_64F, 0, 1, ksize=ksize)
    sobel_mag = np.sqrt(gx ** 2 + gy ** 2)
    sobel_map = normalize_edge_map(sobel_mag)
    t = time.time() - start
    print(f'Time spent: {t:.6f} seconds\n')

    return sobel_map, t

def Canny(img_blur, thresholds=(80, 120)):
    # -------------------------------------
    print('Canny')
    # -------------------------------------

    start = time.time()

    canny_map = cv2.Canny(image=img_blur, threshold1=thresholds[0], threshold2=thresholds[1])
    t = time.time() - start
    print(f'Time spent: {t:.6f} seconds\n')
    return canny_map, t

def LoG(img_gray, ksize, sigma):
    # -------------------------------------
    print('LoG')
    # -------------------------------------

    start = time.time()

    lg_k = log_kernel(ksize=ksize, sigma=sigma)
    log = cv2.filter2D(img_gray, ddepth=cv2.CV_32F, kernel=lg_k)
    log_map = normalize_edge_map(log)
    t = time.time() - start
    print(f'Time spent: {t:.6f} seconds\n')
    return log_map, t

def DoG(img_gray, ksize, sigmas=(1.0, 2.0)):
    # -------------------------------------
    print('DoG')
    # -------------------------------------

    start = time.time()

    g_k1 = gaussian_kernel(ksize=ksize, sigma=sigmas[0])
    g_k2 = gaussian_kernel(ksize=ksize, sigma=sigmas[1])
    blur1 = cv2.filter2D(img_gray, ddepth=cv2.CV_32F, kernel=g_k1)
    blur2 = cv2.filter2D(img_gray, ddepth=cv2.CV_32F, kernel=g_k2)

    dog = blur1.astype('float32') - blur2.astype('float32')
    dog_map = normalize_edge_map(dog)

    t = time.time() - start
    print(f'Time spent: {t:.6f} seconds\n')
    return dog_map, t

def qDoG(img_gray, ksize, sigmas=(1.0, 2.0), q=0.02):
    # -------------------------------------
    print('q-DoG')
    # -------------------------------------

    start = time.time()

    qg_k1 = q_gaussian_kernel(ksize=ksize, sigma=sigmas[0], q=q)
    qg_k2 = q_gaussian_kernel(ksize=ksize, sigma=sigmas[1], q=q)

    qblur1 = cv2.filter2D(img_gray, ddepth=cv2.CV_32F, kernel=qg_k1)
    qblur2 = cv2.filter2D(img_gray, ddepth=cv2.CV_32F, kernel=qg_k2)

    qdog = qblur1.astype('float32') - qblur2.astype('float32')

    qdog_map = normalize_edge_map(qdog)

    t = time.time() - start
    print(f'Time spent: {t:.6f} seconds\n')
    return qdog_map, t


# importing the image
img = cv2.imread("mouse_picture.png")

# printing the image
cv2.imshow("Original", img)

# change the color to a gray scale
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#  blurring the image
img_blur = cv2.GaussianBlur(img_gray, (5, 5), sigmaX=0)

ksize = 5
sigma = 1.0

sobel_t = []
canny_t = []
log_t = []
dog_t = []
qdog_t = []

for i in range(1):
    sobel_map, t = Sobel(img_blur, ksize)
    sobel_t.append(t)
    canny_map, t = Canny(img_blur, thresholds=(80, 120))
    canny_t.append(t)

    log_map, t = LoG(img_gray, ksize, sigma=sigma)
    log_t.append(t)
    dog_map, t = DoG(img_gray, ksize, sigmas=(1.0, 2.0))
    dog_t.append(t)
    qdog_map, t = qDoG(img_gray, ksize, sigmas=(1.0, 2.0), q=0.002)
    qdog_t.append(t)

t_mean, t_std = [], []

t_mean.append(np.mean(sobel_t))
t_std.append(np.std(sobel_t))

t_mean.append(np.mean(canny_t))
t_std.append(np.std(canny_t))

t_mean.append(np.mean(log_t))
t_std.append(np.std(log_t))

t_mean.append(np.mean(dog_t))
t_std.append(np.std(dog_t))

t_mean.append(np.mean(qdog_t))
t_std.append(np.std(qdog_t))

edge_images = [sobel_map, canny_map, log_map, dog_map, qdog_map]
edge_titles = ['Sobel', 'Canny', 'LoG', 'DoG', 'qDoG']


def Edge_images():

    plt.figure(figsize=(10, 10))
    for i in range(len(edge_images)):
        plt.imshow(edge_images[i], cmap='gray')
        plt.title(edge_titles[i])
        plt.axis('off')
        plt.savefig(f'edges/{i+1}_{edge_titles[i]}.png')

Edge_images()
"""plt.figure(figsize=(12, 6))
for i in range(len(edge_images)):
    plt.ylabel("Time [s]")
    plt.scatter(edge_titles[i], t_mean[i], alpha=0.5, label=edge_titles[i], c='blue', s=100)
    plt.errorbar(edge_titles[i], t_mean[i], yerr=t_std[i], fmt='none', capsize=4, ecolor='blue')

    plt.grid()
plt.savefig(f'edges/times.png')
"""