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
    beta = 1.0 / (2 * sigma**2)
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

# importing the image
img = cv2.imread("mouse_picture.png")

# printing the image
cv2.imshow("Original", img)

# change the color to a gray scale
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#  blurring the image
img_blur = cv2.GaussianBlur(img_gray, (3, 3), sigmaX=0)


# Sobel Edge Detection
print('Sobel')
start = time.time()
gx = cv2.Sobel(img_blur, cv2.CV_64F, 1, 0, ksize=7)
gy = cv2.Sobel(img_blur, cv2.CV_64F, 0, 1, ksize=7)
sobel_mag = np.sqrt(gx**2 + gy**2)
sobel = normalize_edge_map(sobel_mag)
print(f'Time spent: {(time.time() - start):.6f} seconds\n')

# Canny Edge Detection
print('Canny')
start = time.time()
canny_map = cv2.Canny(image=img_blur, threshold1=80, threshold2=120)
print(f'Time spent: {(time.time() - start):.6f} seconds\n')

print('LoG')
start = time.time()
lg_k = log_kernel(ksize=7, sigma=1.0)
log = cv2.filter2D(img_gray, ddepth=cv2.CV_32F, kernel=lg_k)
log_map = normalize_edge_map(log)
print(f'Time spent: {(time.time() - start):.6f} seconds\n')

sigma1 = 1
sigma2 = 2

print('DoG')
start = time.time()

g_k1 = gaussian_kernel(ksize=7, sigma=1.0)
g_k2 = gaussian_kernel(ksize=7, sigma=2.0)
blur1 = cv2.filter2D(img_gray, ddepth=cv2.CV_32F, kernel=g_k1)
blur2 = cv2.filter2D(img_gray, ddepth=cv2.CV_32F, kernel=g_k2)

dog = blur1.astype('float32') - blur2.astype('float32')

dog_map = normalize_edge_map(dog)

print(f'Time spent: {(time.time() - start):.6f} seconds\n')

print('q-DoG')
start = time.time()

qg_k1 = q_gaussian_kernel(ksize=7, sigma=1.0, q=0.02)
qg_k2 = q_gaussian_kernel(ksize=7, sigma=2.0, q=0.02)

qblur1 = cv2.filter2D(img_gray, ddepth=cv2.CV_32F, kernel=qg_k1)
qblur2 = cv2.filter2D(img_gray, ddepth=cv2.CV_32F, kernel=qg_k2)

qdog = qblur1.astype('float32') - qblur2.astype('float32')

qdog_map = normalize_edge_map(qdog)

print(f'Time spent: {(time.time() - start):.6f} seconds\n')

# display all the images
titles = ['Original', 'Gray', 'Blurred', 'qGauss']
edge_titles = ['Sobel', 'Canny', 'LoG', 'DoG', 'qDoG']
edge_images = [sobel_map, canny_map, log_map, dog_map, qdog_map]

plt.figure(figsize=(10, 10))
for i in range(len(edge_images)):
    plt.imshow(edge_images[i], cmap='gray')
    plt.title(edge_titles[i])
    plt.axis('off')
    plt.savefig(f'edges/{i+1}_{edge_titles[i]}')
