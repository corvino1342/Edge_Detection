import cv2
import matplotlib.pyplot as plt
import numpy as np
import time
import os

os.makedirs('images', exist_ok=True)
os.makedirs('edges', exist_ok=True)



# importing the image
img = cv2.imread("mouse_picture.png")

# printing the image
cv2.imshow("Original", img)

# change the color to a gray scale
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#  blurring the image
img_blur = cv2.GaussianBlur(img_gray, (3, 3), sigmaX=0)


blurred_images = []
for sigma in sigmas:
    kernel = gaussian_kernel(ksize=ksize, sigma=sigma)
    blurred = cv2.filter2D(img_gray, -1, kernel)
    blurred_images.append(blurred)

plt.figure(figsize=(12, 8))
for i, blurred in enumerate(blurred_images):
    plt.subplot(2, 2, i + 1)
    plt.imshow(blurred, cmap='gray')
    plt.title(f'Blurred with σ = {sigmas[i]}')
    plt.axis('off')
plt.tight_layout()
plt.show()



# blurring the image using q-gaussian kernel
kernel_q = q_gaussian_kernel(ksize=7, q=1.2)
img_qblur = cv2.filter2D(img_gray, -1, kernel_q)

# Sobel Edge Detection
print('Sobel')
start = time.time()
sobel = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=5)
sobel = np.abs(sobel)   # convert to magnitude
print(f'Time spent: {(time.time() - start):.6f} seconds\n')

# Canny Edge Detection
print('Canny')
start = time.time()
canny = cv2.Canny(image=img_blur, threshold1=80, threshold2=120)
print(f'Time spent: {(time.time() - start):.6f} seconds\n')

print('LoG')
start = time.time()
log_kernel = np.array([[0,  0, -1,  0,  0],
                       [0, -1, -2, -1,  0],
                       [-1,-2, 16, -2, -1],
                       [0, -1, -2, -1,  0],
                       [0,  0, -1,  0,  0]], dtype=np.float32)
log = cv2.filter2D(img_gray, -1, log_kernel)
log_norm = cv2.normalize(log, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
print(f'Time spent: {(time.time() - start):.6f} seconds\n')

sigma1 = 1
sigma2 = 2

print('DoG')
start = time.time()
blur1 = cv2.GaussianBlur(img_gray, (3, 3), sigmaX=sigma1)
blur2 = cv2.GaussianBlur(img_gray, (3, 3), sigmaX=sigma2)
dog = blur1.astype('float32') - blur2.astype('float32')

edge_kernel = np.array([[0,  1, 0],
                        [1, -10, 1],
                        [0,  1, 0]], dtype=np.float32)
dog_edges = cv2.filter2D(dog, -1, edge_kernel)
dog_edges = np.abs(dog_edges)   # keep only magnitude
dog_norm = cv2.normalize(dog_edges, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

print(f'Time spent: {(time.time() - start):.6f} seconds\n')

k1 = q_gaussian_kernel(ksize=3, q=1.2)
k2 = q_gaussian_kernel(ksize=5, q=1.2)

print('q-DoG')
start = time.time()
blur1 = cv2.filter2D(img_gray, -1, k1)
blur2 = cv2.filter2D(img_gray, -1, k2)
qdog = blur1.astype('float32') - blur2.astype('float32')
qdog = np.abs(qdog)   # keep only magnitude

qdog_norm = cv2.normalize(qdog, None, 0, 255, cv2.NORM_MINMAX)
qdog_norm = qdog_norm.astype(np.uint8)
print(f'Time spent: {(time.time() - start):.6f} seconds\n')


# display all the images
titles = ['Original', 'Gray', 'Blurred', 'qGauss']
edge_titles = ['Sobel', 'Canny', 'LoG', 'DoG', 'qDoG']
images = [img, img_gray, img_blur, img_qblur]
edge_images = [sobel, canny, log_norm, dog_norm, qdog_norm]

plt.figure(figsize=(10, 10))
for i in range(len(images)):
    if len(images[i].shape) == 2:
        plt.imshow(images[i], cmap='gray')
    else:
        plt.imshow(cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB))
    plt.title(titles[i])
    plt.axis('off')
    plt.savefig(f'images/{i+1}_{titles[i]}')

plt.figure(figsize=(10, 10))
for i in range(len(edge_images)):
    plt.imshow(edge_images[i], cmap='gray')
    plt.title(edge_titles[i])
    plt.axis('off')
    plt.savefig(f'edges/{i+1}_{edge_titles[i]}')
