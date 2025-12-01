"""
Tools Usage Examples
Quick examples demonstrating how to use the tools package
"""

import sys
import cv2
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tools import (
    ImageComparator, ImageAnalyzer,
    BatchProcessor, ResultsAnalyzer,
    ModelManager,
    ImageEnhancer, ImageOptimizer, ColorCorrection
)


def example_1_image_comparison():
    """
    Example 1: Compare original and translated images
    """
    print("\n" + "="*60)
    print("Example 1: Image Comparison")
    print("="*60)
    
    try:
        # Load images (make sure these files exist)
        original = cv2.imread('sample_original.jpg')
        translated = cv2.imread('sample_translated.jpg')
        
        if original is None or translated is None:
            print("❌ Sample images not found")
            print("   Please provide 'sample_original.jpg' and 'sample_translated.jpg'")
            return
        
        # Calculate metrics
        metrics = ImageComparator.get_metrics_dict(original, translated)
        
        print("✅ Metrics:")
        for key, value in metrics.items():
            print(f"   {key.upper()}: {value}")
        
        # Create comparison image
        comparison = ImageComparator.create_comparison_image(
            original, translated, method='horizontal'
        )
        cv2.imwrite('comparison_result.jpg', comparison)
        print("\n✅ Comparison image saved to: comparison_result.jpg")
        
    except Exception as e:
        print(f"❌ Error: {e}")


def example_2_image_analysis():
    """
    Example 2: Analyze image properties and quality
    """
    print("\n" + "="*60)
    print("Example 2: Image Analysis")
    print("="*60)
    
    try:
        image_path = 'sample_original.jpg'
        
        if not Path(image_path).exists():
            print("❌ Sample image not found")
            return
        
        # Get image information
        info = ImageAnalyzer.get_image_info(image_path)
        
        print("✅ Image Information:")
        for key, value in info.items():
            print(f"   {key}: {value}")
        
        # Detect image quality issues
        quality = ImageAnalyzer.detect_image_quality(image_path)
        
        print("\n✅ Quality Assessment:")
        for key, value in quality.items():
            print(f"   {key}: {value}")
        
    except Exception as e:
        print(f"❌ Error: {e}")


def example_3_model_management():
    """
    Example 3: Manage and list available models
    """
    print("\n" + "="*60)
    print("Example 3: Model Management")
    print("="*60)
    
    try:
        # Create model manager
        manager = ModelManager(checkpoint_dir='./checkpoints')
        
        # List all models
        models = manager.list_models()
        
        print(f"✅ Found {len(models)} model(s):")
        for i, model in enumerate(models, 1):
            print(f"\n   Model {i}: {model['name']}")
            print(f"   Size: {model['size_mb']} MB")
            print(f"   Tags: {', '.join(model['tags']) if model['tags'] else 'None'}")
        
        # Compare models if more than one exists
        if len(models) >= 2:
            comparison = manager.compare_models(models[0]['name'], models[1]['name'])
            print(f"\n✅ Model Comparison:")
            print(f"   Size difference: {comparison['size_diff_mb']} MB")
        
    except Exception as e:
        print(f"❌ Error: {e}")


def example_4_image_enhancement():
    """
    Example 4: Enhance image quality
    """
    print("\n" + "="*60)
    print("Example 4: Image Enhancement")
    print("="*60)
    
    try:
        image_path = 'sample_original.jpg'
        
        if not Path(image_path).exists():
            print("❌ Sample image not found")
            return
        
        image = cv2.imread(image_path)
        
        print("✅ Applying enhancements...")
        
        # Enhance contrast
        enhanced_contrast = ImageEnhancer.enhance_contrast(image)
        cv2.imwrite('enhanced_contrast.jpg', enhanced_contrast)
        print("   ✅ Contrast enhancement saved to: enhanced_contrast.jpg")
        
        # Enhance sharpness
        enhanced_sharp = ImageEnhancer.enhance_sharpness(image)
        cv2.imwrite('enhanced_sharpness.jpg', enhanced_sharp)
        print("   ✅ Sharpness enhancement saved to: enhanced_sharpness.jpg")
        
        # Denoise
        denoised = ImageEnhancer.denoise(image)
        cv2.imwrite('denoised.jpg', denoised)
        print("   ✅ Denoised image saved to: denoised.jpg")
        
        # White balance
        white_balanced = ColorCorrection.white_balance(image)
        cv2.imwrite('white_balanced.jpg', white_balanced)
        print("   ✅ White balanced image saved to: white_balanced.jpg")
        
    except Exception as e:
        print(f"❌ Error: {e}")


def example_5_batch_processing():
    """
    Example 5: Batch process images
    """
    print("\n" + "="*60)
    print("Example 5: Batch Processing")
    print("="*60)
    
    try:
        # Create batch processor
        processor = BatchProcessor(output_dir='./batch_results')
        
        # Start batch
        processor.start_batch('demo_batch')
        
        print("✅ Batch processing started")
        
        # Get image files (this would be from a folder)
        # For demo, we'll just show the structure
        print("\n✅ Batch processor is ready:")
        print(f"   Output directory: {processor.output_dir}")
        print(f"   Supported formats: {', '.join(processor.SUPPORTED_FORMATS)}")
        
        # Example: Add some dummy results
        processor.add_result(
            'image1.jpg',
            success=True,
            output_path='./batch_results/image1.jpg',
            metrics={'ssim': 0.95, 'psnr': 42.5}
        )
        
        processor.add_result(
            'image2.jpg',
            success=False,
            error_msg='Processing error'
        )
        
        # End batch and get summary
        batch_log = processor.end_batch()
        
        print("\n✅ Batch Summary:")
        summary = processor.get_batch_summary()
        for key, value in summary.items():
            print(f"   {key}: {value}")
        
        print(f"\n✅ Batch log saved to: {processor.output_dir}/{batch_log['batch_name']}_log.json")
        
    except Exception as e:
        print(f"❌ Error: {e}")


def example_6_image_optimization():
    """
    Example 6: Optimize images for processing
    """
    print("\n" + "="*60)
    print("Example 6: Image Optimization")
    print("="*60)
    
    try:
        image_path = 'sample_original.jpg'
        
        if not Path(image_path).exists():
            print("❌ Sample image not found")
            return
        
        image = cv2.imread(image_path)
        
        print("✅ Applying optimizations...")
        
        # Auto resize
        resized = ImageOptimizer.auto_resize(image, target_size=256)
        cv2.imwrite('resized_256.jpg', resized)
        print("   ✅ Resized image saved to: resized_256.jpg")
        
        # Compress
        compressed = ImageOptimizer.compress_image(image, quality=80)
        cv2.imwrite('compressed_q80.jpg', compressed)
        print("   ✅ Compressed image saved to: compressed_q80.jpg")
        
        # Normalize
        normalized = ImageOptimizer.normalize_image(image)
        print(f"   ✅ Normalized image (range: [0, 1])")
        
        # CLAHE (Contrast Limited Adaptive Histogram Equalization)
        enhanced = ColorCorrection.adaptive_histogram_equalization(image)
        cv2.imwrite('clahe_enhanced.jpg', enhanced)
        print("   ✅ CLAHE enhanced image saved to: clahe_enhanced.jpg")
        
    except Exception as e:
        print(f"❌ Error: {e}")


def main():
    """
    Main function to run all examples
    """
    print("\n" + "🛠️  "*20)
    print("ZSXT Tools Package - Usage Examples")
    print("🛠️  "*20)
    
    print("\n📌 Note: Some examples require sample images:")
    print("   - sample_original.jpg")
    print("   - sample_translated.jpg")
    
    examples = [
        ("Image Comparison", example_1_image_comparison),
        ("Image Analysis", example_2_image_analysis),
        ("Model Management", example_3_model_management),
        ("Image Enhancement", example_4_image_enhancement),
        ("Batch Processing", example_5_batch_processing),
        ("Image Optimization", example_6_image_optimization),
    ]
    
    print("\n📋 Available Examples:")
    for i, (name, _) in enumerate(examples, 1):
        print(f"   {i}. {name}")
    
    print("\n" + "="*60)
    print("Running all examples...")
    print("="*60)
    
    for name, example_func in examples:
        try:
            example_func()
        except Exception as e:
            print(f"\n⚠️  Error in {name}: {e}")
    
    print("\n" + "="*60)
    print("✅ Examples completed!")
    print("="*60)
    print("\n📖 For more details, see TOOLS_GUIDE.md")


if __name__ == '__main__':
    main()
