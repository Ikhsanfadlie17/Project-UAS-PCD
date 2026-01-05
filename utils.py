import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import mean_squared_error as mse

def add_noise(image, noise_type, intensity):
    """Menambahkan noise ke gambar"""
    if noise_type == "Gaussian":
        mean = 0
        sigma = intensity
        gauss = np.random.normal(mean, sigma, image.shape)
        noisy = np.clip(image + gauss, 0, 255).astype(np.uint8)
        return noisy
    
    elif noise_type == "Salt & Pepper":
        noisy = image.copy()
        # Salt
        num_salt = np.ceil(intensity * image.size * 0.01)
        coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape[:2]]
        noisy[coords[0], coords[1]] = 255
        
        # Pepper
        num_pepper = np.ceil(intensity * image.size * 0.01)
        coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape[:2]]
        noisy[coords[0], coords[1]] = 0
        return noisy
    
    elif noise_type == "Speckle":
        gauss = np.random.randn(*image.shape)
        noisy = image + image * gauss * (intensity / 100)
        noisy = np.clip(noisy, 0, 255).astype(np.uint8)
        return noisy
    
    elif noise_type == "Poisson":
        vals = len(np.unique(image))
        vals = 2 ** np.ceil(np.log2(vals))
        noisy = np.random.poisson(image * vals / (intensity + 1)) / float(vals)
        noisy = np.clip(noisy, 0, 255).astype(np.uint8)
        return noisy
    
    return image

def apply_denoising(image, algorithm, params):
    """Aplikasi algoritma denoising"""
    if len(image.shape) == 2:
        gray = image
    else:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    if algorithm == "Gaussian Blur":
        ksize = params.get('kernel_size', 5)
        if ksize % 2 == 0:
            ksize += 1
        result = cv2.GaussianBlur(gray, (ksize, ksize), 0)
    
    elif algorithm == "Median Filter":
        ksize = params.get('kernel_size', 5)
        if ksize % 2 == 0:
            ksize += 1
        result = cv2.medianBlur(gray, ksize)
    
    elif algorithm == "Bilateral Filter":
        d = params.get('d', 9)
        sigma_color = params.get('sigma_color', 75)
        sigma_space = params.get('sigma_space', 75)
        result = cv2.bilateralFilter(gray, d, sigma_color, sigma_space)
    
    elif algorithm == "Non-Local Means":
        h = params.get('h', 10)
        template_window = params.get('template_window', 7)
        search_window = params.get('search_window', 21)
        if len(image.shape) == 3:
            result = cv2.fastNlMeansDenoisingColored(gray, None, h, h, template_window, search_window)
        else:
            result = cv2.fastNlMeansDenoising(gray, None, h, template_window, search_window)
    
    elif algorithm == "Morphological Opening":
        ksize = params.get('kernel_size', 5)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
        result = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
    
    elif algorithm == "Morphological Closing":
        ksize = params.get('kernel_size', 5)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
        result = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
    
    else:
        result = gray
    
    if len(image.shape) == 3 and len(result.shape) == 3:
        result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
    
    return result

def calculate_metrics(original, processed):
    """Hitung metrics quality"""
    if len(original.shape) == 3:
        original_gray = cv2.cvtColor(original, cv2.COLOR_RGB2GRAY)
        processed_gray = cv2.cvtColor(processed, cv2.COLOR_RGB2GRAY)
    else:
        original_gray = original
        processed_gray = processed
    
    psnr_value = psnr(original_gray, processed_gray)
    ssim_value = ssim(original_gray, processed_gray)
    mse_value = mse(original_gray, processed_gray)
    
    return {
        'PSNR': psnr_value,
        'SSIM': ssim_value,
        'MSE': mse_value
    }

def compare_algorithms(image, algorithms, default_params):
    """Compare multiple algorithms"""
    results = {}
    for algo in algorithms:
        denoised = apply_denoising(image, algo, default_params.get(algo, {}))
        results[algo] = denoised
    return results
