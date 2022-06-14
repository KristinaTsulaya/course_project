from matplotlib import pyplot as plt
from skimage.feature import canny
from skimage.filters import threshold_otsu, threshold_sauvola
from skimage.transform import resize, rotate
from skimage.io import imread, imshow
from skimage.color import rgb2gray, rgb2hsv, label2rgb
from skimage.segmentation import slic, mark_boundaries



img_train = imread("/Users/kristina Tsulaya/Downloads/photoeditorsdk-export.png")[:, :, :3]
img_minions = imread("/Users/kristina Tsulaya/Downloads/minions.jpeg")[:, :, :3]
img_bmw = imread("/Users/kristina Tsulaya/Downloads/bmw.jpeg")[:, :, :3]
img_nyny = imread("/Users/kristina Tsulaya/Downloads/nyny.png")[:, :, :3]

#1

plt.figure(figsize=(11, 5))
plt.subplot(121)
imshow(img_train)
hsv_train = rgb2hsv(img_train)
plt.subplot(122)
hsv_train_colorbar = plt.imshow(hsv_train)
plt.colorbar(hsv_train_colorbar, fraction=0.046, pad=0.04)


#2

gray_minions = rgb2gray(img_minions)
plt.figure(figsize=(11, 7))

for i in range(9):

    binarized_gray = (gray_minions > i * 0.1) * 1
    plt.subplot(5, 2, i + 1)
    plt.title("Threshold: >" + str(round(i * 0.1, 1)))
    plt.imshow(binarized_gray, cmap='gray')

imshow(img_minions)
plt.title("original")

#3
plt.figure(figsize=(11, 7))
plt.subplot(121), imshow(img_bmw)
plt.title('Original Image')
plt.subplot(122), imshow(resize(img_bmw, (300, 300)))
plt.title('Resized Image')

#4
plt.figure(figsize=(11, 7))
gray_nyny = rgb2gray(img_nyny)
plt.subplot(2, 2, 1)
plt.title("original")
plt.imshow(img_nyny)
threshold = threshold_otsu(gray_nyny)
binarized_nyny = (gray_nyny > threshold) * 1
plt.subplot(2, 2, 2)
plt.title("Niblack Thresholding")
plt.imshow(binarized_nyny, cmap="gray")
threshold = threshold_sauvola(gray_nyny)
plt.subplot(2, 2, 3)
plt.title("Sauvola Thresholding")
plt.imshow(threshold, cmap="gray")
binarized_nyny = (gray_nyny > threshold) * 1
plt.subplot(2, 2, 4)
plt.title("Sauvola Thresholding - Converting to 0's and 1's")
plt.imshow(binarized_nyny, cmap="gray")

#5
plt.figure(figsize=(11, 7))
bmw_segments = slic(img_bmw, n_segments=50, compactness=10)
plt.subplot(1, 2, 1)
plt.imshow(img_bmw)
plt.title("original")
plt.subplot(1, 2, 2)
plt.imshow(label2rgb(bmw_segments, img_bmw, kind='avg'))

#6
plt.figure(figsize=(11, 7))
plt.subplot(1, 2, 1)
plt.imshow(img_bmw)
plt.subplot(1, 2, 2)
plt.imshow(mark_boundaries(img_bmw, slic(img_bmw, n_segments=10, compactness=10)))

#7
plt.figure(figsize=(11, 7))
plt.subplot(1, 2, 1)
plt.title("the original")
plt.imshow(img_nyny)
plt.subplot(1, 2, 2)
plt.title("edges")
plt.imshow(canny(gray_nyny, sigma=3))

#8
plt.figure(figsize=(11, 7))
plt.subplot(1, 3, 1)
plt.title("the original")
plt.imshow(img_minions)
plt.subplot(1, 3, 2)
plt.title("rotate 180")
plt.imshow(rotate(img_minions, 180))
plt.subplot(1, 3, 3)
plt.title("rotate 45 + back")
plt.imshow(rotate(img_minions, angle=45, resize=True))


plt.show()

