"""
Image Comparison Tool
Provides utilities for comparing original and translated images
"""

import cv2
import numpy as np
from pathlib import Path


class ImageComparator:
    """Compare two images side by side with metrics"""
    
    @staticmethod
    def calculate_mse(img1, img2):
        """Calculate Mean Squared Error between two images"""
        if img1.shape != img2.shape:
            img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
        
        mse = np.mean((img1.astype(float) - img2.astype(float)) ** 2)
        return mse
    
    @staticmethod
    def calculate_ssim(img1, img2):
        """Calculate Structural Similarity Index"""
        if img1.shape != img2.shape:
            img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
        
        # Convert to grayscale if needed
        if len(img1.shape) == 3:
            img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
            img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        else:
            img1_gray = img1
            img2_gray = img2
        
        # Calculate SSIM
        c1 = (0.01 * 255) ** 2
        c2 = (0.03 * 255) ** 2
        
        mean1 = cv2.blur(img1_gray.astype(float), (11, 11))
        mean2 = cv2.blur(img2_gray.astype(float), (11, 11))
        mean1_sq = mean1 ** 2
        mean2_sq = mean2 ** 2
        mean1_mean2 = mean1 * mean2
        
        sigma1_sq = cv2.blur(img1_gray.astype(float) ** 2, (11, 11)) - mean1_sq
        sigma2_sq = cv2.blur(img2_gray.astype(float) ** 2, (11, 11)) - mean2_sq
        sigma12 = cv2.blur(img1_gray.astype(float) * img2_gray.astype(float), (11, 11)) - mean1_mean2
        
        ssim = ((2 * mean1_mean2 + c1) * (2 * sigma12 + c2)) / \
               ((mean1_sq + mean2_sq + c1) * (sigma1_sq + sigma2_sq + c2))
        
        return np.mean(ssim)
    
    @staticmethod
    def calculate_psnr(img1, img2):
        """Calculate Peak Signal-to-Noise Ratio"""
        mse = ImageComparator.calculate_mse(img1, img2)
        if mse == 0:
            return 100  # Perfect match
        
        max_pixel = 255.0
        psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
        return psnr
    
    @staticmethod
    def create_comparison_image(original, translated, method='horizontal'):
        """
        Create a comparison image
        method: 'horizontal', 'vertical', or 'overlay'
        """
        if original.shape != translated.shape:
            translated = cv2.resize(translated, (original.shape[1], original.shape[0]))
        
        if method == 'horizontal':
            comparison = np.hstack([original, translated])
        elif method == 'vertical':
            comparison = np.vstack([original, translated])
        elif method == 'overlay':
            comparison = cv2.addWeighted(original, 0.5, translated, 0.5, 0)
        else:
            comparison = np.hstack([original, translated])
        
        return comparison
    
    @staticmethod
    def get_metrics_dict(img1, img2):
        """Get all metrics as a dictionary"""
        return {
            'mse': round(ImageComparator.calculate_mse(img1, img2), 4),
            'ssim': round(ImageComparator.calculate_ssim(img1, img2), 4),
            'psnr': round(ImageComparator.calculate_psnr(img1, img2), 2)
        }


class ImageAnalyzer:
    """Analyze image properties"""
    
    @staticmethod
    def get_image_info(image_path):
        """Get comprehensive image information"""
        img = cv2.imread(str(image_path))
        if img is None:
            return None
        
        height, width = img.shape[:2]
        channels = img.shape[2] if len(img.shape) == 3 else 1
        
        # Calculate file size
        file_size = Path(image_path).stat().st_size
        file_size_mb = file_size / (1024 * 1024)
        
        # Calculate histogram
        if channels == 3:
            colors = ['blue', 'green', 'red']
            hist_values = {}
            for i, color in enumerate(colors):
                hist = cv2.calcHist([img], [i], None, [256], [0, 256])
                hist_values[color] = int(np.mean(hist))
        else:
            hist = cv2.calcHist([img], [0], None, [256], [0, 256])
            hist_values = {'gray': int(np.mean(hist))}
        
        return {
            'width': width,
            'height': height,
            'channels': channels,
            'file_size_mb': round(file_size_mb, 2),
            'pixels': width * height,
            'aspect_ratio': round(width / height, 2),
            'histogram': hist_values
        }
    
    @staticmethod
    def detect_image_quality(image_path):
        """Detect image quality issues"""
        img = cv2.imread(str(image_path))
        if img is None:
            return None
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Calculate Laplacian variance (blur detection)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        variance = laplacian.var()
        
        # Determine blur status
        if variance < 100:
            blur_status = "Very Blurry"
            blur_score = "Low (< 100)"
        elif variance < 500:
            blur_status = "Blurry"
            blur_score = "Medium (100-500)"
        else:
            blur_status = "Clear"
            blur_score = "High (> 500)"
        
        # Calculate brightness
        brightness = np.mean(gray)
        if brightness < 50:
            brightness_status = "Too Dark"
        elif brightness > 200:
            brightness_status = "Too Bright"
        else:
            brightness_status = "Normal"
        
        return {
            'blur_status': blur_status,
            'blur_score': blur_score,
            'brightness': round(brightness, 1),
            'brightness_status': brightness_status,
            'laplacian_variance': round(variance, 2)
        }


if __name__ == '__main__':
    # Example usage
    print("Image Comparison Tool")
    print("=" * 50)
    
    # Example: Compare two images
    # comp = ImageComparator()
    # metrics = comp.get_metrics_dict(img1, img2)
    # print(f"Metrics: {metrics}")
    
    # Example: Analyze image
    # analyzer = ImageAnalyzer()
    # info = analyzer.get_image_info('sample.jpg')
    # print(f"Info: {info}")
