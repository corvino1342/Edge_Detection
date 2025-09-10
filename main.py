import cv2
import matplotlib.pyplot as plt

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
edges = cv2.Canny(image=img_blur, threshold1=100, threshold2=200) # Canny Edge Detection

# display all the images
titles = ['Original', 'Gray', 'Blurred', 'Sobel X', 'Sobel Y', 'Sobel XY', 'Canny']
images = [img, img_gray, img_blur, sobelx, sobely, sobelxy, edges]

plt.figure(figsize=(15, 5))
for i in range(len(images)):
    plt.subplot(1, len(images), i+1)
    if len(images[i].shape) == 2:
        plt.imshow(images[i], cmap='gray')
    else:
        plt.imshow(cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB))
    plt.title(titles[i])
    plt.axis('off')
plt.show()