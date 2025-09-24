import matplotlib.pyplot as plt
import numpy as np
import math

def gaussian_kernel(ksize=3, sigma=1.0):
    """Generate a 2D Gaussian kernel."""
    assert ksize % 2 == 1
    half = ksize // 2
    x = np.arange(-half, half + 1)
    y = np.arange(-half, half + 1)
    X, Y = np.meshgrid(x, y)
    kernel = np.exp(-(X**2 + Y**2) / (2 * sigma**2))
    kernel /= kernel.sum()
    return kernel

def q_gaussian_kernel(ksize = 5, sigma = 1.0, q = 1.0):
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
    y = np.arange(-half, half + 1)
    X, Y = np.meshgrid(x, y)
    R2 = X**2 + Y**2
    beta = (1 / (2 * sigma ** 2)) # this is the scale parameter, based on the value of Gaussian sigma
    if q == 1:
        kernel = np.exp(-beta * R2)
    else:
        kernel = (1 - (1 - q) * beta * R2) ** (1 / (1 - q))
        kernel[kernel < 0] = 0
    kernel /= kernel.sum()
    return kernel.astype(np.float32)



sigmas = np.linspace(0.1, 5, 16)
ksize = 15

plt.figure(figsize=(12, 8))
for i, sigma in enumerate(sigmas):

    #kernel = gaussian_kernel(ksize, sigma)
    kernel = q_gaussian_kernel(ksize, sigma, q=0.9)


    plt.subplot(math.ceil(len(sigmas)/int(math.sqrt(len(sigmas)))), math.ceil(len(sigmas)/int(math.sqrt(len(sigmas)))), i + 1)

    plt.imshow(kernel, cmap='viridis')
    plt.title(f'σ = {sigma:.3f}')
    plt.axis('off')
plt.suptitle(f'Gaussian Kernels (ksize={ksize})')
plt.tight_layout()
plt.show()



