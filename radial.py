# _____________________________________________________________________________
# radial.py - Fast Radial Symmetry Transform.
# _____________________________________________________________________________
# -----------------------------------------------------------------------------
# G. Loy and A. Zelinsky, “A Fast Radial Symmetry Transform for Detecting
# Points of Interest,” Computer Vision — ECCV 2002 Lecture Notes in Computer
# Science, pp. 358–368, 2002.
# -----------------------------------------------------------------------------
import cv2
import numpy
# -----------------------------------------------------------------------------
# Functions
# -----------------------------------------------------------------------------


def grad(image):
    if len(image.shape) == 2:
        gray = image
    else:
        image = cv2.GaussianBlur(image, (3, 3), 0)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Gradient-X
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    # Gradient-Y
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    return grad_x, grad_y


def radial(image, radius, alpha=3, beta=0.2, std=0.5):
    # Get image gradients.
    grad_y, grad_x = grad(image)
    row = grad_x.shape[0]
    col = grad_x.shape[1]
    temp_size = (row + (2 * radius), col + (2 * radius), 1)
    # Calculate gradient magnitudes.
    grad_mags = numpy.sqrt(numpy.add(numpy.multiply(grad_x, grad_x),
                                     numpy.multiply(grad_y, grad_y)))
    # Calculate gradient threshold for improved performance.
    gthresh = numpy.amax(grad_mags)*beta
    # orientation projection image
    ort_proj_im = numpy.zeros(temp_size, numpy.float64)
    # magnitude projection image
    mag_proj_im = numpy.zeros(temp_size, numpy.float64)
    for i in range(row):
        for j in range(col):
            grx = grad_x[i][j]
            gry = grad_y[i][j]
            grad_mag = grad_mags[i][j]
            if grad_mag > gthresh:
                t_gx = int(numpy.round((grx / grad_mag) * radius))
                t_gy = int(numpy.round((gry / grad_mag) * radius))
                # Calculate the positively-affected pixel
                idx_pos = (i + t_gx + radius, j + t_gy + radius)
                # Calculate the negatively-affected pixel
                idx_neg = (i - t_gx + radius, j - t_gy + radius)
                # Full(bright and dark) transform
                ort_proj_im[idx_pos] += 1
                ort_proj_im[idx_neg] -= 1
                mag_proj_im[idx_pos] += grad_mag
                mag_proj_im[idx_neg] -= grad_mag
    # Calculate O_n_hat
    t_opim = numpy.abs(ort_proj_im)
    t_opim = t_opim / numpy.amax(t_opim)
    t_opim = numpy.abs(t_opim)
    # Calculate M_n_hat
    t_mpim = numpy.abs(mag_proj_im)
    t_mpim = mag_proj_im / numpy.amax(t_mpim)

    # Calculate F_n
    f = numpy.multiply((numpy.power(t_opim, alpha)), t_mpim)
    kernel_size = int(numpy.ceil(radius / 2))
    if kernel_size % 2 == 0:
        kernel_size += 1
    # Calculate S_n
    s = cv2.GaussianBlur(f, (kernel_size, kernel_size), (std * radius))
    return s[radius:row + radius, radius:col + radius]


# -----------------------------------------------------------------------------
# End of radial.py
# -----------------------------------------------------------------------------
# _____________________________________________________________________________
# TITLE - radial.py
# AUTHOR - Ogulcan Ozer.
# C_DATE - 3 JUN 2020
# U_DATE - 14 JUN 2020
# _____________________________________________________________________________
