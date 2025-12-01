"""
Advanced Preprocessing Toolkit
Enhanced image preprocessing utilities including augmentation and optimization
"""

import cv2
import numpy as np
from pathlib import Path


class ImageEnhancer:
    """Enhance image quality"""
    
    @staticmethod
    def denoise(image, method='bilateral'):
        """Denoise image"""
        if method == 'bilateral':
            return cv2.bilateralFilter(image, 9, 75, 75)
        elif method == 'gaussian':
            return cv2.GaussianBlur(image, (5, 5), 0)
        elif method == 'morphological':
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
        return image
    
    @staticmethod
    def enhance_contrast(image, clip_limit=2.0):
        """Enhance image contrast using CLAHE"""
        if len(image.shape) == 3:
            # Convert to LAB and apply CLAHE to L channel
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
            lab[:, :, 0] = clahe.apply(lab[:, :, 0])
            return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        else:
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
            return clahe.apply(image)
    
    @staticmethod
    def enhance_sharpness(image, strength=1.5):
        """Enhance image sharpness"""
        kernel = np.array([[-1, -1, -1],
                          [-1,  9, -1],
                          [-1, -1, -1]]) / 9.0
        
        sharpened = cv2.filter2D(image, -1, kernel)
        return cv2.addWeighted(image, 1 - strength/10, sharpened, strength/10, 0)
    
    @staticmethod
    def adjust_brightness(image, value=30):
        """Adjust image brightness"""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(float)
        hsv[:, :, 2] = hsv[:, :, 2] + value
        hsv[:, :, 2] = np.clip(hsv[:, :, 2], 0, 255)
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    
    @staticmethod
    def adjust_saturation(image, value=1.2):
        """Adjust image saturation"""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(float)
        hsv[:, :, 1] = hsv[:, :, 1] * value
        hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


class ImageAugmenter:
    """Apply augmentation to images"""
    
    @staticmethod
    def rotate(image, angle=15):
        """Rotate image"""
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(image, matrix, (w, h))
    
    @staticmethod
    def flip(image, direction='horizontal'):
        """Flip image"""
        if direction == 'horizontal':
            return cv2.flip(image, 1)
        elif direction == 'vertical':
            return cv2.flip(image, 0)
        elif direction == 'both':
            return cv2.flip(image, -1)
        return image
    
    @staticmethod
    def perspective_transform(image, scale=0.2):
        """Apply perspective transformation"""
        h, w = image.shape[:2]
        pts1 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
        
        offset = int(h * scale)
        pts2 = np.float32([[offset, 0], [w-offset, offset], 
                          [0, h-offset], [w-offset, h-offset]])
        
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        return cv2.warpPerspective(image, matrix, (w, h))
    
    @staticmethod
    def elastic_transform(image, alpha=34, sigma=4):
        """Apply elastic deformation"""
        h, w = image.shape[:2]
        dx = np.random.randn(h, w) * sigma
        dy = np.random.randn(h, w) * sigma
        
        x, y = np.meshgrid(np.arange(w), np.arange(h))
        indices = y + dy.astype(int), x + dx.astype(int)
        
        indices = (np.clip(indices[0], 0, h-1), np.clip(indices[1], 0, w-1))
        
        if len(image.shape) == 3:
            return image[indices]
        return image[indices]


class ImageOptimizer:
    """Optimize images for processing"""
    
    @staticmethod
    def auto_resize(image, target_size=256, preserve_aspect=True):
        """Automatically resize image"""
        h, w = image.shape[:2]
        
        if preserve_aspect:
            scale = target_size / max(h, w)
            new_w = int(w * scale)
            new_h = int(h * scale)
            resized = cv2.resize(image, (new_w, new_h))
            
            # Add padding
            canvas = np.zeros((target_size, target_size, 3), dtype=np.uint8)
            offset_h = (target_size - new_h) // 2
            offset_w = (target_size - new_w) // 2
            canvas[offset_h:offset_h+new_h, offset_w:offset_w+new_w] = resized
            return canvas
        else:
            return cv2.resize(image, (target_size, target_size))
    
    @staticmethod
    def compress_image(image, quality=90):
        """Compress image to reduce file size"""
        _, compressed = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, quality])
        return cv2.imdecode(compressed, cv2.IMREAD_COLOR)
    
    @staticmethod
    def normalize_image(image):
        """Normalize image values to [0, 1]"""
        return image.astype(float) / 255.0
    
    @staticmethod
    def standardize_image(image):
        """Standardize image values (zero mean, unit variance)"""
        normalized = ImageOptimizer.normalize_image(image)
        mean = normalized.mean(axis=(0, 1), keepdims=True)
        std = normalized.std(axis=(0, 1), keepdims=True)
        return (normalized - mean) / (std + 1e-7)


class ColorCorrection:
    """Color correction utilities"""
    
    @staticmethod
    def white_balance(image, reference_color=None):
        """Apply white balance"""
        result = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(float)
        
        avg_a = np.mean(result[:, :, 1])
        avg_b = np.mean(result[:, :, 2])
        
        result[:, :, 1] = result[:, :, 1] - ((avg_a - 128) * 1.5)
        result[:, :, 2] = result[:, :, 2] - ((avg_b - 128) * 1.5)
        
        result = np.clip(result, 0, 255)
        return cv2.cvtColor(result.astype(np.uint8), cv2.COLOR_LAB2BGR)
    
    @staticmethod
    def apply_lut(image, lut_table):
        """Apply Look-up Table for color correction"""
        return cv2.LUT(image, lut_table)
    
    @staticmethod
    def histogram_equalization(image):
        """Equalize histogram"""
        if len(image.shape) == 3:
            # Apply to each channel
            for i in range(3):
                image[:, :, i] = cv2.equalizeHist(image[:, :, i])
            return image
        else:
            return cv2.equalizeHist(image)
    
    @staticmethod
    def adaptive_histogram_equalization(image):
        """Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)"""
        if len(image.shape) == 3:
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            l = clahe.apply(l)
            
            lab = cv2.merge([l, a, b])
            return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        else:
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            return clahe.apply(image)


class EdgeDetection:
    """Edge detection utilities"""
    
    @staticmethod
    def canny_edge(image, threshold1=100, threshold2=200):
        """Apply Canny edge detection"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        return cv2.Canny(gray, threshold1, threshold2)
    
    @staticmethod
    def sobel_edge(image):
        """Apply Sobel edge detection"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
        
        magnitude = np.sqrt(sobelx**2 + sobely**2)
        magnitude = (magnitude / magnitude.max() * 255).astype(np.uint8)
        
        return magnitude
    
    @staticmethod
    def laplacian_edge(image):
        """Apply Laplacian edge detection"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        return cv2.Laplacian(gray, cv2.CV_64F).astype(np.uint8)


if __name__ == '__main__':
    print("Advanced Preprocessing Toolkit")
    print("=" * 50)
    
    # Example usage
    # enhancer = ImageEnhancer()
    # image = cv2.imread('sample.jpg')
    # enhanced = enhancer.enhance_contrast(image)
    # cv2.imwrite('enhanced.jpg', enhanced)
