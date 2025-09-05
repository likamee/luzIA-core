import os
import random
import shutil

import cv2
import numpy as np
import skimage
import skimage.feature
from PIL import Image, ImageEnhance, ImageFilter
from skimage.filters import gaussian


def process_filters(source, FILENAME, method, pato=""):
    """Process the image with the selected filter
    :param source: path to the source folder
    :param FILENAME: name of the file to be processed
    :param method: filter to be applied
    :param pato: pathology to be trained
    :return: processed image
    """

    TYPE = "alteradas" if pato else "normais"
    FOLDER = pato if pato else "/" + TYPE

    """ if method == 'raw':
        shutil.copy(os.path.join(source+TYPE+pato, FILENAME), source+'algo'+FOLDER)
    else:       """

    # IMAGEGS = cv2.imread(os.path.join(source+TYPE+pato, FILENAME), 0)
    image = cv2.imread(os.path.join(source + TYPE + pato, FILENAME))
    # IMAGEGS = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # IMAGEGS = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if check_bright(image):  # or check_blurry(image):
        cv2.imwrite(source + "excluidas/" + FILENAME, image)
        return False
    image = globals()[method](image)
    # image = globals()['unsharp'](image)

    cv2.imwrite(source + "algo" + FOLDER + "/" + FILENAME, image)
    # image.save(source+'algo'+FOLDER+'/'+FILENAME)

    return True
    # image.save(source+'algo/normais/'+FILENAME)
    # shutil.copy(os.path.join(source+'normais', FILENAME), source+'algo/normais')


def check_blurry(image):
    # compute the Laplacian of the image and then return the focus
    # measure, which is simply the variance of the Laplacian
    threshold = 150
    fm = cv2.Laplacian(image, cv2.CV_64F).var()
    return fm < threshold


def check_bright(image):
    thresholds = [135, 35]  # 135 35
    is_light = np.mean(image) > thresholds[0]
    is_dark = np.mean(image) < thresholds[1]
    return is_light or is_dark


def combo_filter(image):
    R, image, B = cv2.split(image)
    CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))  # 2 #8
    image = CLAHE.apply(image)
    image = cv2.bitwise_not(image)
    return image


def reddit_constrast(image):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)  # convert from BGR to LAB color space
    l, a, b = cv2.split(lab)  # split on 3 different channels
    l2 = clahe.apply(l)  # apply CLAHE to the L-channel
    lab = cv2.merge((l2, a, b))  # merge channels
    image = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)  # convert from LAB to BGR
    return image


def gaussian_blur(image):
    # image = np.array(img)
    image_blur = cv2.GaussianBlur(image, (65, 65), 10)
    # new_image = cv2.subtract(img,image_blur).astype('float32') # WRONG, the result is not stored in float32 directly
    new_image = cv2.subtract(image, image_blur, dtype=cv2.CV_32F)
    image = cv2.normalize(
        new_image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U
    )
    return image


def hsv(image):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    return image


def canny(image):
    image = skimage.feature.canny(
        image=image,
        sigma=3,
        low_threshold=2,
        high_threshold=10,
    )
    image = image * 255
    return image


def contrast(image):
    image = Image.fromarray(image.astype("uint8"))
    converter = ImageEnhance.Contrast(image)
    image = converter.enhance(1.5)
    return np.array(image)


def saturation(image):
    image = Image.fromarray(image.astype("uint8"))
    converter = ImageEnhance.Color(image)
    image = converter.enhance(1.5)
    return np.array(image)


def sharpness(image):
    image = Image.fromarray(image.astype("uint8"))
    converter = ImageEnhance.Sharpness(image)
    image = converter.enhance(1.5)
    return np.array(image)


def brightness(image):
    image = Image.fromarray(image.astype("uint8"))
    converter = ImageEnhance.Brightness(image)
    image = converter.enhance(1.5)
    return np.array(image)


def conservative_smoothing(image):
    filter_size = 9

    data = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    new_image = data.copy()

    temp = []
    indexer = filter_size // 2
    new_image = data.copy()
    nrow, ncol = data.shape
    for i in range(nrow):
        for j in range(ncol):
            for k in range(i - indexer, i + indexer + 1):
                for m in range(j - indexer, j + indexer + 1):
                    if (k > -1) and (k < nrow):
                        if (m > -1) and (m < ncol):
                            temp.append(data[k, m])
            temp.remove(data[i, j])
            max_value = max(temp)
            min_value = min(temp)
            if data[i, j] > max_value:
                new_image[i, j] = max_value
            elif data[i, j] < min_value:
                new_image[i, j] = min_value
            temp = []

    return new_image


def crimmins(image):
    data = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    new_image = data.copy()

    nrow = len(data)
    ncol = len(data[0])

    # Dark pixel adjustment

    # First Step
    # N-S
    for i in range(1, nrow):
        for j in range(ncol):
            if data[i - 1, j] >= (data[i, j] + 2):
                new_image[i, j] += 1
    data = new_image
    # E-W
    for i in range(nrow):
        for j in range(ncol - 1):
            if data[i, j + 1] >= (data[i, j] + 2):
                new_image[i, j] += 1
    data = new_image
    # NW-SE
    for i in range(1, nrow):
        for j in range(1, ncol):
            if data[i - 1, j - 1] >= (data[i, j] + 2):
                new_image[i, j] += 1
    data = new_image
    # NE-SW
    for i in range(1, nrow):
        for j in range(ncol - 1):
            if data[i - 1, j + 1] >= (data[i, j] + 2):
                new_image[i, j] += 1
    data = new_image
    # Second Step
    # N-S
    for i in range(1, nrow - 1):
        for j in range(ncol):
            if (data[i - 1, j] > data[i, j]) and (data[i, j] <= data[i + 1, j]):
                new_image[i, j] += 1
    data = new_image
    # E-W
    for i in range(nrow):
        for j in range(1, ncol - 1):
            if (data[i, j + 1] > data[i, j]) and (data[i, j] <= data[i, j - 1]):
                new_image[i, j] += 1
    data = new_image
    # NW-SE
    for i in range(1, nrow - 1):
        for j in range(1, ncol - 1):
            if (data[i - 1, j - 1] > data[i, j]) and (data[i, j] <= data[i + 1, j + 1]):
                new_image[i, j] += 1
    data = new_image
    # NE-SW
    for i in range(1, nrow - 1):
        for j in range(1, ncol - 1):
            if (data[i - 1, j + 1] > data[i, j]) and (data[i, j] <= data[i + 1, j - 1]):
                new_image[i, j] += 1
    data = new_image
    # Third Step
    # N-S
    for i in range(1, nrow - 1):
        for j in range(ncol):
            if (data[i + 1, j] > data[i, j]) and (data[i, j] <= data[i - 1, j]):
                new_image[i, j] += 1
    data = new_image
    # E-W
    for i in range(nrow):
        for j in range(1, ncol - 1):
            if (data[i, j - 1] > data[i, j]) and (data[i, j] <= data[i, j + 1]):
                new_image[i, j] += 1
    data = new_image
    # NW-SE
    for i in range(1, nrow - 1):
        for j in range(1, ncol - 1):
            if (data[i + 1, j + 1] > data[i, j]) and (data[i, j] <= data[i - 1, j - 1]):
                new_image[i, j] += 1
    data = new_image
    # NE-SW
    for i in range(1, nrow - 1):
        for j in range(1, ncol - 1):
            if (data[i + 1, j - 1] > data[i, j]) and (data[i, j] <= data[i - 1, j + 1]):
                new_image[i, j] += 1
    data = new_image
    # Fourth Step
    # N-S
    for i in range(nrow - 1):
        for j in range(ncol):
            if data[i + 1, j] >= (data[i, j] + 2):
                new_image[i, j] += 1
    data = new_image
    # E-W
    for i in range(nrow):
        for j in range(1, ncol):
            if data[i, j - 1] >= (data[i, j] + 2):
                new_image[i, j] += 1
    data = new_image
    # NW-SE
    for i in range(nrow - 1):
        for j in range(ncol - 1):
            if data[i + 1, j + 1] >= (data[i, j] + 2):
                new_image[i, j] += 1
    data = new_image
    # NE-SW
    for i in range(nrow - 1):
        for j in range(1, ncol):
            if data[i + 1, j - 1] >= (data[i, j] + 2):
                new_image[i, j] += 1
    data = new_image

    # Light pixel adjustment

    # First Step
    # N-S
    for i in range(1, nrow):
        for j in range(ncol):
            if data[i - 1, j] <= (data[i, j] - 2):
                new_image[i, j] -= 1
    data = new_image
    # E-W
    for i in range(nrow):
        for j in range(ncol - 1):
            if data[i, j + 1] <= (data[i, j] - 2):
                new_image[i, j] -= 1
    data = new_image
    # NW-SE
    for i in range(1, nrow):
        for j in range(1, ncol):
            if data[i - 1, j - 1] <= (data[i, j] - 2):
                new_image[i, j] -= 1
    data = new_image
    # NE-SW
    for i in range(1, nrow):
        for j in range(ncol - 1):
            if data[i - 1, j + 1] <= (data[i, j] - 2):
                new_image[i, j] -= 1
    data = new_image
    # Second Step
    # N-S
    for i in range(1, nrow - 1):
        for j in range(ncol):
            if (data[i - 1, j] < data[i, j]) and (data[i, j] >= data[i + 1, j]):
                new_image[i, j] -= 1
    data = new_image
    # E-W
    for i in range(nrow):
        for j in range(1, ncol - 1):
            if (data[i, j + 1] < data[i, j]) and (data[i, j] >= data[i, j - 1]):
                new_image[i, j] -= 1
    data = new_image
    # NW-SE
    for i in range(1, nrow - 1):
        for j in range(1, ncol - 1):
            if (data[i - 1, j - 1] < data[i, j]) and (data[i, j] >= data[i + 1, j + 1]):
                new_image[i, j] -= 1
    data = new_image
    # NE-SW
    for i in range(1, nrow - 1):
        for j in range(1, ncol - 1):
            if (data[i - 1, j + 1] < data[i, j]) and (data[i, j] >= data[i + 1, j - 1]):
                new_image[i, j] -= 1
    data = new_image
    # Third Step
    # N-S
    for i in range(1, nrow - 1):
        for j in range(ncol):
            if (data[i + 1, j] < data[i, j]) and (data[i, j] >= data[i - 1, j]):
                new_image[i, j] -= 1
    data = new_image
    # E-W
    for i in range(nrow):
        for j in range(1, ncol - 1):
            if (data[i, j - 1] < data[i, j]) and (data[i, j] >= data[i, j + 1]):
                new_image[i, j] -= 1
    data = new_image
    # NW-SE
    for i in range(1, nrow - 1):
        for j in range(1, ncol - 1):
            if (data[i + 1, j + 1] < data[i, j]) and (data[i, j] >= data[i - 1, j - 1]):
                new_image[i, j] -= 1
    data = new_image
    # NE-SW
    for i in range(1, nrow - 1):
        for j in range(1, ncol - 1):
            if (data[i + 1, j - 1] < data[i, j]) and (data[i, j] >= data[i - 1, j + 1]):
                new_image[i, j] -= 1
    data = new_image
    # Fourth Step
    # N-S
    for i in range(nrow - 1):
        for j in range(ncol):
            if data[i + 1, j] <= (data[i, j] - 2):
                new_image[i, j] -= 1
    data = new_image
    # E-W
    for i in range(nrow):
        for j in range(1, ncol):
            if data[i, j - 1] <= (data[i, j] - 2):
                new_image[i, j] -= 1
    data = new_image
    # NW-SE
    for i in range(nrow - 1):
        for j in range(ncol - 1):
            if data[i + 1, j + 1] <= (data[i, j] - 2):
                new_image[i, j] -= 1
    data = new_image
    # NE-SW
    for i in range(nrow - 1):
        for j in range(1, ncol):
            if data[i + 1, j - 1] <= (data[i, j] - 2):
                new_image[i, j] -= 1
    data = new_image

    return data


def unsharp(image):
    image = Image.fromarray(image.astype("uint8"))
    new_image = image.filter(ImageFilter.UnsharpMask(radius=2, percent=150))

    return new_image


def raw(image):
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def low_pass(image):
    # image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV) # convert to HSV
    image2 = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    dft = cv2.dft(np.float32(image2), flags=cv2.DFT_COMPLEX_OUTPUT)

    # shift the zero-frequncy component to the center of the spectrum
    dft_shift = np.fft.fftshift(dft)
    rows, cols = image2.shape
    crow, ccol = rows // 2, cols // 2

    # create a mask first, center square is 1, remaining all zeros
    mask = np.zeros((rows, cols, 2), np.uint8)
    mask[crow - 30 : crow + 30, ccol - 30 : ccol + 30] = 1

    # apply mask and inverse DFT
    fshift = dft_shift * mask
    f_ishift = np.fft.ifftshift(fshift)
    img_back = cv2.idft(f_ishift)
    img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])

    return img_back


def high_pass(image):
    # image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV) # convert to HSV
    image2 = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    dft = cv2.dft(np.float32(image2), flags=cv2.DFT_COMPLEX_OUTPUT)

    # shift the zero-frequncy component to the center of the spectrum
    dft_shift = np.fft.fftshift(dft)
    rows, cols = image2.shape
    crow, ccol = rows // 2, cols // 2

    # create a mask first, center square is 1, remaining all zeros
    mask = np.zeros((rows, cols, 2), np.uint8)
    mask[crow - 30 : crow + 30, ccol - 30 : ccol + 30] = 1

    # apply mask and inverse DFT
    fshift = dft_shift * mask
    f_ishift = np.fft.ifftshift(fshift)
    img_back = cv2.idft(f_ishift)
    img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])

    return img_back


# for FILENAME in random.sample(filesn, len(filesn)):
def apply_filter(method, pato, source, filesn, filesp, prop, hq, lq, type_img):
    count = 0
    reso = hq if type_img == "h" else lq

    count = 0
    for filename in filesp:
        if any(list(map(lambda x: x in filename, reso))):
            if not process_filters(source, filename, method, "/" + pato):
                continue
        count = count + 1

    n_pato = count
    # imgn = int(n_pato*prop)
    count = 0
    for filename in filesn:
        if any(list(map(lambda x: x in filename, reso))):
            if not process_filters(source, filename, method):
                continue
        count = count + 1


"""if ("20sus" not in FILENAME and "021sus" not in FILENAME and "60sus" not in FILENAME and "70sus" not in FILENAME and "80sus" not in FILENAME):
    continue"""
