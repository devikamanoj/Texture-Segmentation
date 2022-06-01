import time
import numpy as np
from numpy import linalg as nplinalg
from scipy import linalg as sclinalg
from scipy import ndimage
import matplotlib.pyplot as plt
import math
from skimage import io, color
from skimage.color import rgb2gray

def getspecHist(image, ws, nbins=11):
    #image: Image
    #ws: half window size
    #nbins: number of bins of histograms (fixed)
    
    #Histograms provide a feature vector that gives a global impression of the image
    #In this approach, we use integral histograms for each pixel, to compute their spectral histograms
    h, w, bn = image.shape
    
    #Computing bin ids for integral histogram approach
    for i in range(bn):
        b_max = np.max(image[:, :, i])
        b_min = np.min(image[:, :, i])
        b_interval = (b_max - b_min) * 1. / nbins
        image[:, :, i] = np.floor((image[:, :, i] - b_min) / b_interval) #Assigning the bin id for each pixel

    image[image >= nbins] = nbins-1
    image = np.int32(image)

    #Convert bin ids to one_hot_encoding representation (histogram for each pixel)
    one_hot_pix = []
    for i in range(bn):
        one_hot_pix_b = np.zeros((h*w, nbins), dtype=np.int32)
        one_hot_pix_b[np.arange(h*w), image[:, :, i].flatten()] = 1
        one_hot_pix.append(one_hot_pix_b.reshape((h, w, nbins)))

    #Compute integral histograms (by iteratively adding histograms for each pixel)
    integral_hist = np.concatenate(one_hot_pix, axis=2)

    np.cumsum(integral_hist, axis=1, out=integral_hist, dtype=np.float32) #left to right
    np.cumsum(integral_hist, axis=0, out=integral_hist, dtype=np.float32) #top to bottom

    #Compute local histograms by using integral histograms at 4 corners
    #Pad histograms to handle local windows near image borders
    padding_l = np.zeros((h, ws + 1, nbins * bn), dtype=np.int32)
    padding_r = np.tile(integral_hist[:, -1:, :], (1, ws, 1))
    integral_hist_pad_tmp = np.concatenate([padding_l, integral_hist, padding_r], axis=1) #horizontal padding
    padding_t = np.zeros((ws + 1, integral_hist_pad_tmp.shape[1], nbins * bn), dtype=np.int32)
    padding_b = np.tile(integral_hist_pad_tmp[-1:, :, :], (ws, 1, 1))
    integral_hist_pad = np.concatenate([padding_t, integral_hist_pad_tmp, padding_b], axis=0) #vertical padding

    #Find the sub-arrays for each of the 4 corners of the image
    integral_hist_1 = integral_hist_pad[ws + 1 + ws:, ws + 1 + ws:, :]
    integral_hist_2 = integral_hist_pad[:-ws - ws - 1, :-ws - ws - 1, :]
    integral_hist_3 = integral_hist_pad[ws + 1 + ws:, :-ws - ws -1, :]
    integral_hist_4 = integral_hist_pad[:-ws - ws - 1, ws + 1 + ws:, :]
    
    #Compute spectral histogram based on integral histogram
    specHist = integral_hist_1 + integral_hist_2 - integral_hist_3 - integral_hist_4
    histsum = np.sum(specHist, axis=-1, keepdims=True) * 1. / bn
    specHist = np.float32(specHist) / np.float32(histsum)

    #Return an array containing the local spectral histogram at each pixel
    return specHist

def factorizedSeg(image, ws, omega):

    #image: n-band image
    #ws: window size for local special histogram
    #omega: error threshod for estimating segment number. need to adjust for different filter bank.

    #Compute dimensions of image
    N1, N2, bn = image.shape

    ws = ws // 2

    #Get the spectral histogram for each pixel of the image, and its dimensions
    specHist = getspecHist(image, ws)
    sh_dim = specHist.shape[2]

    #Apply SVD to obtain factored matrices with low rank
    Z = (specHist.reshape((N1 * N2, sh_dim)))
    u, v = nplinalg.eig(np.dot(Z.T, Z))
    u_sorted = np.sort(u)
    idx = np.argsort(u)
    k = np.abs(u_sorted)

    #Estimate the number of segments from singular values (Effective rank of feature matrix)
    lse_ratio = np.cumsum(k) * 1. / (N1 * N2)
    segn = np.sum(lse_ratio > omega)
    print("Estimated number of segments: {}".format(segn))

    if segn <= 1:
        segn = 2

    dimn = segn
    beta = v[:, idx[-1:-dimn-1:-1]]

    #Project features onto the subspace
    Y = np.dot(Z, beta)

    #Remove all features with high edgeness (using formula)
    h, w, _ = Y.reshape((N1, N2, dimn)).shape
    edge_map = np.ones((h, w)) * -1
    for i in range(ws, h-ws-1):
        for j in range(ws, w-ws-1):
            edge_map[i, j] = np.sqrt(np.sum((specHist[i - ws, j, :] - specHist[i + ws, j, :])**2) + np.sum((specHist[i, j - ws, :] - specHist[i, j + ws, :])**2))
    
    edge_map_flatten = edge_map.flatten()
    Y_new = Y[(edge_map_flatten >= 0) & (edge_map_flatten <= np.max(edge_map)*0.4), :]

    #Find representative features using a variation of k-means clustering
    centroid = np.zeros((segn, dimn), dtype=np.float32)
    L = np.sum(Y_new ** 2, axis=1)

    #Find ei (feature with max length)
    centroid[0, :] = Y_new[np.argmax(L), :]
    D = np.sum((centroid[0, :] - Y_new) ** 2, axis=1)
    #Find ej (feature with max distance to set)
    centroid[1, :] = Y_new[np.argmax(D), :]

    #Select points near simplex vertices using formula
    centroid_id = 1
    while centroid_id < segn-1:
        centroid_id += 1
        D_tmp = np.zeros((centroid_id, Y_new.shape[0]), dtype=np.float32)
        for i in range(centroid_id):
            D_tmp[i, :] = np.sum((centroid[i, :] - Y_new) ** 2, axis=1)
        D = np.min(D_tmp, axis=0)
        centroid[centroid_id, :] = Y_new[np.argmax(D), :]

    D_cen2all = np.zeros((segn, Y_new.shape[0]), dtype=np.float32)
    centroid_new = np.zeros((segn, dimn), dtype=np.float32)

    #Perform k-means (k = no of segments)
    converging = True
    while converging:
        for i in range(segn):
            D_cen2all[i, :] = np.sum((centroid[i, :] - Y_new) ** 2, axis=1)

        cls_id = np.argmin(D_cen2all, axis=0)

        for i in range(segn):
            centroid_new[i, :] = np.mean(Y_new[cls_id == i, :], axis=0)

        if np.max((centroid_new - centroid)**2) < .00001:
            converging = False
        else:
            centroid = centroid_new * 1.
    centroid_new = centroid_new.T

    ZZTinv = sclinalg.inv(np.dot(centroid_new.T, centroid_new))
    Beta = np.dot(np.dot(ZZTinv, centroid_new.T), Y.T)

    seg_label = np.argmax(Beta, axis=0)

    #Finding non-negative constraints
    w0 = np.dot(beta, centroid_new)
    dnorm0 = 1

    #Apply ALS (Alternating least squares) algorithm ->
    #Compute final matrix using least squares method (in alternating fashion)
    h = Beta * 1.
    for i in range(100):
        tmp, _, _, _ = nplinalg.lstsq(np.dot(w0.T, w0) + np.eye(segn) * .01, np.dot(w0.T, Z.T))
        h = np.maximum(0, tmp)
        tmp, _, _, _ = nplinalg.lstsq(np.dot(h, h.T) + np.eye(segn) * .01, np.dot(h, Z))
        w = np.maximum(0, tmp)
        w = w.T * 1.

        d = Z.T - np.dot(w, h)
        dnorm = np.sqrt(np.mean(d * d))
        if np.abs(dnorm - dnorm0) < .1:
            break

        w0 = w * 1.
        dnorm0 = dnorm * 1.

    seg_label = np.argmax(h, axis=0)

    #Return the final result after segmentation
    return seg_label.reshape((N1, N2))

def log_filter(sgm, fsize):
    #Laplacian of Gaussian filter
    #sgm: sigma in Gaussian
    #fsize: filter size, [h, w]

    wins_x = fsize[1] // 2
    wins_y = fsize[0] // 2

    out = np.zeros(fsize, dtype=np.float32)

    for x in range(-wins_x, wins_x+1):
        for y in range(-wins_y, wins_y+1):
            out[wins_y+y, wins_x+x] = - 1. / (math.pi * sgm**4.) * (1. - (x*x+y*y)/(2.*sgm*sgm)) * math.exp(-(x*x+y*y)/(2.*sgm*sgm))

    return out-np.mean(out)


def gabor_filter(sgm, theta):
    #Gabor filter
    #sgm: sigma in Gaussian
    #theta: direction

    phs=0
    gamma=1
    wins=int(math.floor(sgm*2))
    f=1/(sgm*2.)
    out=np.zeros((2*wins+1, 2*wins+1))

    for x in range(-wins, wins+1):
        for y in range(-wins, wins+1):
            xPrime = x * math.cos(theta) + y * math.sin(theta)
            yPrime = y * math.cos(theta) - x * math.sin(theta)
            out[wins+y, wins+x] = 1/(2*math.pi*sgm*sgm)*math.exp(-.5*((xPrime)**2+(yPrime*gamma)**2)/sgm**2)*math.cos(2*math.pi*f*xPrime+phs)
    return out-np.mean(out)


def filterImage(img, filter_list):
    imgs = []
    #Perform filtering based on filters provided as args
    for filter in filter_list:
        if filter[0] == 'log':
            f = log_filter(filter[1], filter[2])
            tmp = ndimage.correlate(np.float32(img), f, mode='reflect')
            imgs.append(tmp)

        elif filter[0] == 'gabor':
            f = gabor_filter(filter[1], filter[2])
            tmp = ndimage.correlate(np.float32(img), f, mode='reflect')
            imgs.append(tmp)

    return np.float32(np.stack(imgs, axis=2))

if __name__ == '__main__':
    start_time = time.time()
    #Convert image to greyscale

    img = rgb2gray(io.imread(r'tm1_1_1_trial.png'))
    
    filter_list = [('log', .5, [3, 3]), ('log', 1, [5, 5]),
                   ('gabor', 1.5, 0), ('gabor', 1.5, math.pi/2), ('gabor', 1.5, math.pi/4), ('gabor', 1.5, -math.pi/4),
                   ('gabor', 2.5, 0), ('gabor', 2.5, math.pi/2), ('gabor', 2.5, math.pi/4), ('gabor', 2.5, -math.pi/4)
                   ]

    filter_out = filterImage(img, filter_list=filter_list)

    #Include the original image as a band
    image = np.concatenate((np.float32(img.reshape((img.shape[0], img.shape[1], 1))), filter_out), axis=2)
    output = factorizedSeg(image, ws=25, omega=.045)

    print('Factorized segmentation runs in {} seconds.'.format(time.time() - start_time))

    #Plot results
    fig, ax = plt.subplots(ncols=2, figsize=(12, 6))
    ax[0].set_title('Initial image (Converted to greyscale)')
    ax[1].set_title('Final image after segmentation')
    plt.setp(ax, xticks=[], yticks=[])
    ax[0].imshow(img, cmap='gray')
    ax[1].imshow(output, cmap='gray')
    plt.show()