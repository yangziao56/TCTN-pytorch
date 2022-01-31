import numpy as np

def pixel_to_dBZ_2(img):
    return img * 70.0

def dBZ_to_pixel_2(dBZ_img):
    return np.clip(dBZ_img / 70.0, a_min=0.0, a_max=1.0)


def pixel_to_dBZ(img):
    """

    Parameters
    ----------
    img : np.ndarray or float

    Returns
    -------

    """
    return img * 70.0 - 10.0

def dBZ_to_pixel(dBZ_img):
    """

    Parameters
    ----------
    dBZ_img : np.ndarray

    Returns
    -------

    """
    return np.clip((dBZ_img + 10.0) / 70.0, a_min=0.0, a_max=1.0)


def pixel_to_rainfall(img, a=58.53, b=1.56):
    """Convert the pixel values to real rainfall intensity

    Parameters
    ----------
    img : np.ndarray
    a : float32, optional
    b : float32, optional

    Returns
    -------
    rainfall_intensity : np.ndarray
    """
    dBZ = pixel_to_dBZ(img)
    dBR = (dBZ - 10.0 * np.log10(a)) / b
    rainfall_intensity = np.power(10, dBR / 10.0)
    return rainfall_intensity


def rainfall_to_pixel(rainfall_intensity, a=58.53, b=1.56):
    """Convert the rainfall intensity to pixel values

    Parameters
    ----------
    rainfall_intensity : np.ndarray
    a : float32, optional
    b : float32, optional

    Returns
    -------
    pixel_vals : np.ndarray
    """
    dBR = np.log10(rainfall_intensity) * 10.0
    # dBZ = 10b log(R) +10log(a)
    dBZ = dBR * b + 10.0 * np.log10(a)
    pixel_vals = (dBZ + 10.0) / 70.0
    return pixel_vals

def dBZ_to_rainfall(dBZ, a=58.53, b=1.56):
    return np.power(10, (dBZ - 10 * np.log10(a))/(10*b))

def rainfall_to_dBZ(rainfall, a=58.53, b=1.56):
    return 10*np.log10(a) + 10*b*np.log10(rainfall)

