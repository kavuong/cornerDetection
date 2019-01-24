# cornerDetection

rgb2gray() converts a color image to grayscale using the y component of the RGB to YIQ conversion. It iterates
through each pixel of the grayscale image and multiples each color channel value by its corresponding coefficient
in the y component.

smooth1D() takes a grayscale image and smooths it in one direction to remove noise using a programmed Gaussian filter.
The kernel size is appropriately calculated using the sigma value and the Gaussian filter equation provided in class.
A filter consisting of integers from -100 to 100 is produced, and then each value is multiplied by e^(-(x^2/2 * sigma^2))
to produce a Gaussian filter. The actual filter consists of all the values greater than 1/1000th of the peak value (1),
which is accounted for in the actual filter. The proper 1D filter consists of the values from the (center - kernelSize)
index to the (center + kernelSize) index, since the kernel size is in one direction.
This filter is then used for convolution. The filter is applied to the image using the convolve1d function, and that
result is normalized by dividing each image element by the sum of the elements in the filter. The border is handled by
finding the specific filter elements involved in convolution (i.e. a partial filter) with the index of the border
elements and then dividing the border elements by the sum of just the specific filter elements. A smoothed image in 1D
is returned.

smooth2D() takes a grayscale image and smooths it in two directions by convolving it with two 1D filters.
The image is smoothed in the vertical direction first by taking the transpose of the image and passing it into the
smooth1D function. That result is then transposed again and passed into the smooth1D function once more to smooth the
image in the horizontal direction. The result is then returned.

harris() implements Harris's corner detection algorithm in Python. The images of Ix and Iy were calculated using finite
differences with the pixel values of the grayscale image. Ix^2, Iy^2 and IxIy were calculated by matrix multiplication.
Those images were smoothed in 2D using the smooth2D() function.
The R image was calculated by generating a 2D array of pixels with the same dimensions as the image and initializing
each array element with its appropriate R value based on the Ix^2, Iy^2 and IxIy values of the corresponding pixel in
those images.
For calculating potential corners, I created a helper function that would generate an array of values of all of a
pixel's 8-neighbors. I iterated through each pixel of the R image and compared its value to that of each of its
8-neighbors, and if the pixel value was indeed greater than each of the 8-neighbors' values, then the pixel's
coordinates will be appended to an array that stores all corner candidates.
Then for each corner candidate, I computed the cornerness value by quadratic approximation up to sub-pixel accuracy, and
disqualified candidates whose cornerness did not reach the threshold.
