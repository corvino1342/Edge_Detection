import cv2
import matplotlib.pyplot as plt
import numpy as np


def q_gaussian_kernel(ksize = 5, beta = 1.0, q = 1.0):
    """
    Generate a 2D q-Gaussian kernel.

    Args:
        ksize (int): kernel size (odd number, e.g. 3, 5, 7).
        beta (float): scale parameter (like 1/(2*sigma^2) for Gaussian).
        q (float): entropic index (q=1 → normal Gaussian).

    Returns:
        kernel (numpy.ndarray): normalized q-Gaussian kernel.
    """
    assert ksize % 2 == 1
    half = ksize // 2

    x = np.arange(-half, half + 1)
    y = np.arange(-half, half + 1)

    X, Y = np.meshgrid(x, y)
    R2 = X**2 + Y**2

    if q == 0:
        kernel = np.exp(-beta * R2)
    else:
        kernel = (1 - (1 - q) * beta * R2) ** (1 / (1 - q))
        kernel[kernel < 0] = 0

    kernel /= kernel.sum()
    return kernel.astype(np.float32)

# importing the image
img = cv2.imread("mouse_picture.png")

# printing the image
cv2.imshow("Original", img)

# change the color to a gray scale
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#  blurring the image
img_blur = cv2.GaussianBlur(img_gray, (3, 3), sigmaX=0, sigmaY=0)
cv2.imshow("Blurred", img_blur)

# Sobel Edge Detection
sobelx = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=5)
sobely = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=5)
sobelxy = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=5)

# Canny Edge Detection
edges = cv2.Canny(image=img_blur, threshold1=20, threshold2=100)

log = cv2.Laplacian(img_blur, ddepth=cv2.CV_64F)

sigma1 = 1
sigma2 = 2
blur1 = cv2.GaussianBlur(img_gray, (0, 0), sigmaX=sigma1)
blur2 = cv2.GaussianBlur(img_gray, (0, 0), sigmaX=sigma2)
dog = blur1.astype('float32') - blur2.astype('float32')

kernel_q = q_gaussian_kernel(ksize=7, beta=0.05, q=1.2)

img_qblur = cv2.filter2D(img_gray, -1, kernel_q)

k1 = q_gaussian_kernel(ksize=3, beta=0.05, q=1.1)
k2 = q_gaussian_kernel(ksize=9, beta=0.05, q=1.1)

blur1 = cv2.filter2D(img_gray, -1, k1)
blur2 = cv2.filter2D(img_gray, -1, k2)

qdog = blur1.astype('float32') - blur2.astype('float32')

# display all the images
titles = ['Original', 'Gray', 'Blurred', 'qGauss']
edge_titles = ['SobelX', 'SobelY', 'SobelXY', 'Canny', 'LoG', 'DoG', 'qDoG']
images = [img, img_gray, img_blur, img_qblur]
edge_images = [sobelx, sobely, sobelxy, edges, log, dog, qdog]

plt.figure(figsize=(10, 10))
for i in range(len(images)):
    if len(images[i].shape) == 2:
        plt.imshow(images[i], cmap='gray')
    else:
        plt.imshow(cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB))
    plt.title(titles[i])
    plt.axis('off')
    plt.savefig(f'images/{i}_{titles[i]}')

plt.figure(figsize=(10, 10))
for i in range(len(edge_images)):
    if len(edge_images[i].shape) == 2:
        plt.imshow(edge_images[i], cmap='gray')
    else:
        plt.imshow(cv2.cvtColor(edge_images[i], cv2.COLOR_BGR2RGB))
    plt.title(edge_titles[i])
    plt.axis('off')
    plt.savefig(f'edges/{i}_{edge_titles[i]}')