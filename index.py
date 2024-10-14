import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

def add_gaussian_noise(image, mean=0, std=0.06*255):
    gaussian_noise = np.random.normal(mean, std, image.shape).astype(np.float32)
    noisy_image = cv2.add(image.astype(np.float32), gaussian_noise)
    noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)  # Ensure pixel values are valid
    return noisy_image

def apply_gaussian_convolution(image, kernel_size=(5, 5), sigma=1.0):
    return cv2.GaussianBlur(image, kernel_size, sigma)

def apply_clahe(image):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(image)

def calculate_psnr(original, processed):
    mse = np.mean((original - processed) ** 2)
    if mse == 0:
        return float('inf')  # PSNR is infinite if no difference
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr

def calculate_mse(original, processed):
    return np.mean((original - processed) ** 2)

def process_images_in_folder(folder_path):    
    for filename in os.listdir(folder_path):
        if filename.endswith(('.png', '.jpg', '.jpeg')):  
            image_path = os.path.join(folder_path, filename)
            
            # 1. Load Dataset (Chest X-Ray Viral Pneumonia)
            image = cv2.imread(image_path)
            
            if image is None:
                print(f"Failed to load {filename}")
                continue

            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            noisy_image = add_gaussian_noise(gray_image)
            blurred_image = apply_gaussian_convolution(noisy_image)
            clahe_image = apply_clahe(blurred_image)
            gray_original_resized = cv2.resize(gray_image, (clahe_image.shape[1], clahe_image.shape[0]))

            psnr_value = calculate_psnr(gray_original_resized, clahe_image)
            mse_value = calculate_mse(gray_original_resized, clahe_image)

            print(f"Image: {filename} - PSNR: {psnr_value:.2f} dB, MSE: {mse_value:.2f}")
            
            plt.figure(figsize=(15, 5))
            plt.subplots_adjust(wspace=0.4, hspace=0.4)
            
            plt.subplot(2, 4, 1)
            plt.title(f'Original Image - {filename}')
            plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            plt.axis('off')

            plt.subplot(2, 4, 2)
            plt.title('Grayscale + Gaussian Noise 0.6')
            plt.imshow(noisy_image, cmap='gray')
            plt.axis('off')

            plt.subplot(2, 4, 3)
            plt.title('Gaussian Convolution')
            plt.imshow(blurred_image, cmap='gray')
            plt.axis('off')
            
            plt.subplot(2, 4, 4)
            plt.title('Processed Image (CLAHE)')
            plt.imshow(clahe_image, cmap='gray')
            plt.axis('off')
            
            plt.subplot(2, 4, 5)
            plt.title('Histogram of Original Image')
            plt.hist(image.ravel(), bins=256, range=[0, 256])
            plt.xlabel('Pixel Intensity')
            plt.ylabel('Frequency')
            
            plt.subplot(2, 4, 6)
            plt.title('Histogram of Noisy Image')
            plt.hist(noisy_image.ravel(), bins=256, range=[0, 256])
            plt.xlabel('Pixel Intensity')
            plt.ylabel('Frequency')
            
            plt.subplot(2, 4, 7)
            plt.title('Histogram of Gaussian Convolution')
            plt.hist(blurred_image.ravel(), bins=256, range=[0, 256])
            plt.xlabel('Pixel Intensity')
            plt.ylabel('Frequency')
            
            plt.subplot(2, 4, 8)
            plt.title('Histogram of CLAHE')
            plt.hist(clahe_image.ravel(), bins=256, range=[0, 256])
            plt.xlabel('Pixel Intensity')
            plt.ylabel('Frequency')

            plt.show()

folder_path = 'pneumonia'
process_images_in_folder(folder_path)