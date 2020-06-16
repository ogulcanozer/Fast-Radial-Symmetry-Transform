import cv2
import numpy
from radial import frst

image = cv2.imread(".\\images\\lenna.jpg")
image_out = numpy.zeros((image.shape[0], image.shape[1]))

# Calculate and add for S_n = 3, 5.
for i in range(3, 7, 2):
    # By default alpha = 2, beta = %20 and std. factor = 0.5
    image_out = image_out + frst(image, i)

# Normalize the output image to grayscale.
image_out = (image_out - numpy.amin(image_out)) / (
    numpy.amax(image_out) - numpy.amin(image_out))
image_out *= 255

cv2.imwrite(".\\images\\lennaFRST.jpg", image_out)
