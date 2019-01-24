################################################################################
# Corner detection
################################################################################
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.ndimage import convolve1d
from scipy.ndimage import gaussian_filter


################################################################################
#  perform RGB to grayscale conversion
################################################################################
def rgb2gray(img_color) :
    # input:
    #    img_color - a h x w x 3 numpy ndarray (dtype = np.unit8) holding
    #                the color image
    # return:
    #    img_gray - a h x w numpy ndarray (dtype = np.float64) holding
    #               the grayscale image

    # TODO: using the Y channel of the YIQ model to perform the conversion
    num_rows = img_color.shape[0]
    num_cols = img_color.shape[1]
    img_gray = np.zeros(shape=(num_rows,num_cols), dtype= np.float64)

    for x in range(0, num_rows):
        for y in range(0, num_cols):
            img_gray[x,y] =  0.299 * img_color[x, y, 0] + 0.587 * img_color[x, y, 1] + 0.114 * img_color[x, y, 1]

    return img_gray

################################################################################
#  perform 1D smoothing using a 1D horizontal Gaussian filter
################################################################################
def smooth1D(img, sigma) :
    # input :
    #    img - a h x w numpy ndarray holding the image to be smoothed
    #    sigma - sigma value of the 1D Gaussian function
    # return:
    #    img_smoothed - a h x w numpy ndarry holding the 1D smoothing result


    # TODO: form a 1D horizontal Guassian filter of an appropriate size

    h = len(img)
    w = len(img[0])

    n = 100
    x = np.arange(-1 * n, n + 1)
    # solving for x from kernel size equation given in class
    kernelSize = math.ceil(math.sqrt(-2 * sigma ** 2 * math.log(0.001)))
    subFilter = np.exp(-(x ** 2) / (2 * sigma ** 2))
    bottomIndex = int(100 - kernelSize)
    topIndex = int(100 + kernelSize)
    realFilter = subFilter[bottomIndex:topIndex]

    # TODO: convolve the 1D filter with the image;
    #       apply partial filter for the image border

    result = convolve1d(img, realFilter, mode='constant', cval=0.0)
    filterLength = len(realFilter)
    midFilterIndex = math.trunc(filterLength / 2)
    distToMidFromLeft = midFilterIndex
    distToMidFromRight = filterLength - midFilterIndex - 1

    img_smoothed = np.zeros((h, w))

    for row in range(0, h):
        for col in range(0, w):
            # left edge
            if col < distToMidFromLeft:
                numFilterElements = filterLength - midFilterIndex + col
                img_smoothed[row, col] = result[row, col] / sum(realFilter[-numFilterElements:])

            # right edge
            elif w - col - 1 < distToMidFromRight:
                numFilterElements = w - col + distToMidFromLeft
                img_smoothed[row, col] = result[row, col] / sum(realFilter[:numFilterElements])

            else:
                img_smoothed[row, col] = result[row, col] / sum(realFilter)

    return img_smoothed

################################################################################
#  perform 2D smoothing using 1D convolutions
################################################################################
def smooth2D(img, sigma) :
    # input:
    #    img - a h x w numpy ndarray holding the image to be smoothed
    #    sigma - sigma value of the Gaussian function
    # return:
    #    img_smoothed - a h x w numpy array holding the 2D smoothing result


    img_vert = img.T

    # TODO: smooth the image along the vertical direction

    img_vert = smooth1D(img_vert, sigma)

    # TODO: smooth the image along the horizontal direction

    img_horz = img_vert.T

    img_smoothed = smooth1D(img_horz, sigma)

    return img_smoothed

################################################################################
#   perform Harris corner detection
################################################################################
# helper function for returning 8-neighbors
def neighbors(arr, row, col):
    for rowNum in row-1, row, row+1:
        if rowNum < 0 or rowNum == len(arr): continue
        for colNum in col-1, col, col+1:
            if colNum < 0 or colNum == len(arr[rowNum]): continue
            if rowNum == row and colNum == col: continue
            yield arr[rowNum, colNum]


def harris(img, sigma, threshold) :
    # input:
    #    img - a h x w numpy ndarry holding the input image
    #    sigma - sigma value of the Gaussian function used in smoothing
    #    threshold - threshold value used for identifying corners
    # return:
    #    corners - a list of tuples (x, y, r) specifying the coordinates
    #              (up to sub-pixel accuracy) and cornerness value of each corner

    h = len(img)
    w = len(img[0])

    # TODO: compute Ix & Iy
    ix = np.zeros((h, w))
    iy = np.zeros((h, w))

    for x in range(0, h):
        for y in range(0, w):
            if y == 0:
                ix[x, y] = img[x, 1] - img[x, y]
            elif y == w - 1:
                ix[x, y] = img[x, y] - img[x, y - 1]
            else:
                ix[x, y] = (img[x, y + 1] - img[x, y - 1]) / 2

            if x == 0:
                iy[x, y] = img[1, y] - img[x, y]
            elif x == h - 1:
                iy[x, y] = img[x, y] - img[x - 1, y]
            else:
                iy[x, y] = (img[x + 1, y] - img[x - 1, y]) / 2

    # TODO: compute Ix2, Iy2 and IxIy

    ix2 = ix * ix
    iy2 = iy * iy
    ixiy = ix * iy

    # TODO: smooth the squared derivatives

    ix2 = smooth2D(ix2, sigma)
    iy2 = smooth2D(iy2, sigma)
    ixiy = smooth2D(ixiy, sigma)

    # TODO: compute cornesness functoin R
    r = np.zeros((h, w))
    for x in range(0, h):
        for y in range(0, w):
            r[x, y] = (ix2[x, y] * iy2[x, y] - (ixiy[x, y] ** 2)) - 0.04 * ((ix2[x, y] + iy2[x, y]) ** 2)

    # TODO: mark local maxima as corner candidates;
    #       perform quadratic approximation to local corners upto sub-pixel accuracy
    cornerCandidates = []

    for x in range(0, h):
        for y in range(0, w):
            isPossCorner = True
            for neighborR in neighbors(r, x, y):
                if r[x, y] > neighborR:
                    continue
                else:
                    isPossCorner = False
                    break
            if isPossCorner:
                cornerCandidates.append((x, y))

    # TODO: perform thresholding and discard weak corners

    corners = []

    for candidate in cornerCandidates:
        x = candidate[0]
        y = candidate[1]
        # NEXT TIME: HANDLE POSSIBLE OUT OF ARRAY ERRORS

        if x - 1 < 0 or x - 1 >= h:
            top = 0
        else:
            top = r[x - 1, y]

        if x + 1 < 0 or x + 1 >= h:
            bottom = 0
        else:
            bottom = r[x + 1, y]

        pixel = r[x, y]

        if y + 1 < 0 or y + 1 >= w:
            right = 0
        else:
            right = r[x, y + 1]

        if y - 1 < 0 or y - 1 >= w:
            left = 0
        else:
            left = r[x, y - 1]

        a = (left + right - 2 * pixel) / 2
        b = (bottom + top - 2 * pixel) / 2
        c = (right - left) / 2
        d = (top - bottom) / 2
        e = pixel

        subX = -c / (2 * a)
        subY = -d / (2 * b)

        cornerness = a * subY ** 2 + b * subX ** 2 + c * subY + d * subX + e

        if cornerness > threshold:
            corners.append((y + subY, x + subX, cornerness))

    return sorted(corners, key = lambda corner : corner[2], reverse = True)

################################################################################
#   save corners to a file
################################################################################
def save(outputfile, corners) :
    try :
        file = open(outputfile, 'w')
        file.write('%d\n' % len(corners))
        for corner in corners :
            file.write('%.4f %.4f %.4f\n' % corner)
        file.close()
    except :
        print('Error occurs in writting output to \'%s\''  % outputfile)
        sys.exit(1)

################################################################################
#   load corners from a file
################################################################################
def load(inputfile) :
    try :
        file = open(inputfile, 'r')
        line = file.readline()
        nc = int(line.strip())
        print('loading %d corners' % nc)
        corners = list()
        for i in range(nc) :
            line = file.readline()
            (x, y, r) = line.split()
            corners.append((float(x), float(y), float(r)))
        file.close()
        return corners
    except :
        print('Error occurs in writting output to \'%s\''  % outputfile)
        sys.exit(1)

################################################################################
## main
################################################################################
def main() :
    # REMOVE:
    parser = argparse.ArgumentParser(description = 'COMP3317 Assignment 2')
    parser.add_argument('-i', '--inputfile', type = str, default = 'grid1.jpg', help = 'filename of input image')
    parser.add_argument('-s', '--sigma', type = float, default = 1.0, help = 'sigma value for Gaussain filter')
    parser.add_argument('-t', '--threshold', type = float, default = 1e6, help = 'threshold value for corner detection')
    parser.add_argument('-o', '--outputfile', type = str, help = 'filename for outputting corner detection result')
    args = parser.parse_args()

    print('------------------------------')
    print('COMP3317 Assignment 2')
    print('input file : %s' % args.inputfile)
    print('sigma      : %.2f' % args.sigma)
    print('threshold  : %.2e' % args.threshold)
    print('output file: %s' % args.outputfile)
    print('------------------------------')

    # load the image
    try :
        #img_color = imageio.imread(args.inputfile)
        img_color = plt.imread(args.inputfile)
        print('%s loaded...' % args.inputfile)
    except :
        print('Cannot open \'%s\'.' % args.inputfile)
        sys.exit(1)
    # uncomment the following 2 lines to show the color image
    #plt.imshow(np.uint8(img_color))
    #plt.show()

    # perform RGB to gray conversion
    print('perform RGB to grayscale conversion...')
    img_gray = rgb2gray(img_color)
    # uncomment the following 2 lines to show the grayscale image
    #plt.imshow(np.float32(img_gray), cmap = 'gray')
    #plt.show()

    img_smooth = smooth2D(img_gray, 10)


    plt.figure()
    plt.imshow(np.float32(img_smooth), cmap = 'gray')
    plt.show()

    # perform corner detection
    print('perform Harris corner detection...')
    corners = harris(img_gray, args.sigma, args.threshold)
    plt.imshow(np.float32(img_gray), cmap = 'gray')
    plt.show()


    # plot the corners
    print('%d corners detected...' % len(corners))
    x = [corner[0] for corner in corners]
    y = [corner[1] for corner in corners]
    fig = plt.figure()
    plt.imshow(np.float32(img_gray), cmap = 'gray')
    plt.plot(x, y,'r+',markersize = 5)
    plt.show()

    # save corners to a file
    if args.outputfile :
        save(args.outputfile, corners)
        print('corners saved to \'%s\'...' % args.outputfile)


if __name__ == '__main__':
    main()
