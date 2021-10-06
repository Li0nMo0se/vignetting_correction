import numpy as np
from scipy.ndimage import gaussian_filter


def log_entropy(image, sigma_smooth):
    N = 256
    n_hist = np.zeros(N, dtype=float)

    rescaled_image = (N - 1) * (image / image.max())
    li = (N - 1) * np.log2(1 + rescaled_image) / np.log2(N)
    li_floor = np.floor(li).astype(int)
    li_ceil = np.ceil(li).astype(int)

    np.add.at(n_hist, li_floor, 1 + li_floor - li)
    np.add.at(n_hist, li_ceil, li_ceil - li)

    n_hist_smooth = gaussian_filter(n_hist, sigma_smooth)

    pk = n_hist_smooth / np.sum(n_hist_smooth)
    log_pk = np.where(pk != 0, np.log2(pk), 0)

    return - np.sum(pk * log_pk)


def compute_center_of_mass(image):
    (sum, i_sum, j_sum) = (0, 0, 0)

    for j in range(image.shape[0]):
        for i in range(image.shape[1]):
            sum += image[j, i]

            i_sum += (i + 1) * image[j, i]
            j_sum += (j + 1) * image[j, i]

    return i_sum / sum, j_sum / sum


def compute_r_matrix(image):
    (i_mid, j_mid) = compute_center_of_mass(image)

    res = np.empty_like(image, dtype=float)

    d = np.sqrt(i_mid ** 2 + j_mid ** 2)

    for j in range(image.shape[0]):
        for i in range(image.shape[1]):
            res[j, i] = np.sqrt((i - i_mid) ** 2 + (j - j_mid) ** 2) / d

    return res


def g(r, a, b, c):
    return 1 + a * r ** 2 + b * r ** 4 + c * r ** 6


def verify_constraints(a, b, c):
    if a > 0 and b == 0 and c == 0:
        return True
    if a >= 0 and b > 0 and c == 0:
        return True
    if c == 0 and b < 0 and -a <= 2 * b:
        return True
    if c > 0 and b ** 2 < 3 * a * c:
        return True
    if c > 0 and b ** 2 == 3 * a * c and b >= 0:
        return True
    if c > 0 and b ** 2 == 3 * a * c and -b >= 3 * c:
        return True
    if c == 0:
        return False
    q_p = (-2 * b + np.sqrt(4 * b ** 2 - 12 * a * c)) / (6 * c)
    if c > 0 and b ** 2 > 3 * a * c and q_p <= 0:
        return True
    q_d = (-2 * b - np.sqrt(4 * b ** 2 - 12 * a * c)) / (6 * c)
    if c > 0 and b ** 2 > 3 * a * c and q_d >= 1:
        return True
    if c < 0 and b ** 2 > 3 * a * c and q_p >= 1 and q_d <= 0:
        return True

    return False


def rgb_to_luminance(image):
    return 0.2126 * image[..., 0] + 0.7152 * image[..., 1] + 0.0722 * image[..., 2]


def correct_vignetting(image, sigma_smooth):
    """
    Apply vignetting correction

    Parameters
    ----------
    image : np.arrray
        The input image to be corrected
    sigma_smooth: float
        The sigma smooth used by the gaussian kernel to compute the
        log-intensity entropy

    Returns
    -------
    np.array
        The corrected image (array like `image`)
    """
    if image.ndim != 3:
        raise ValueError(f"Image must be a 3D array (2D colored image). Got {image.ndim}D")

    if sigma_smooth is None:
        sigma_smooth = 2.25  # Default value
    if sigma_smooth < 0:
        raise ValueError(f"Sigma smooth must be >= 0. Got {sigma_smooth}")

    gray_image = rgb_to_luminance(image)

    (a, b, c) = (0.0, 0.0, 0.0)
    delta = 8.0
    h_min = log_entropy(gray_image, sigma_smooth)
    r = compute_r_matrix(gray_image)

    while delta > 1 / 256:
        v_arr = np.array([(a + delta, b, c), (a - delta, b, c),
                          (a, b + delta, c), (a, b - delta, c),
                          (a, b, c + delta), (a, b, c - delta)])

        for v in v_arr:
            if verify_constraints(*v):
                h_tmp = log_entropy(gray_image * g(r, *v), sigma_smooth)
                if h_tmp < h_min:
                    h_min = h_tmp
                    (a, b, c) = v
                    delta = 16.0

        delta /= 2.0

    res = image * np.stack(3 * [g(r, a, b, c)], axis=2)
    return np.clip(res, 0, 255).astype(np.uint8)
