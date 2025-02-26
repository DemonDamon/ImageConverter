#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Common Usage Examples of ImageConverter

This example demonstrates various common use cases of the ImageConverter:
- Loading and resizing images to specific dimensions
- Converting between different formats
- Processing numpy arrays
- Working with URL images
- Using PyTorch tensors
- Base64 encoding/decoding
"""

import os
import sys
import numpy as np
from PIL import Image

# Add parent directory to path so we can import the library
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from image_converter import ImageConverter

def show_examples():
    """Demonstrate various usage examples of ImageConverter"""
    
    # Create processor instance
    processor = ImageConverter()
    
    print("===== Image Processor Usage Examples =====")
    
    # Example 1: Load image and resize to specified resolution
    print("\nExample 1: Load image and resize to specified resolution")
    try:
        # Load image and resize to fixed resolution (224x224)
        image = processor.process_image(
            'sample.jpg',  # Replace with your image path
            output_type='pil',
            min_size=224,  # Minimum size
            max_size=224   # Setting max size same as min size forces square output
        )
        print(f"Resized image dimensions: {image.size}")
        
        # Save the resized image
        processor.save_image(image, 'output_224x224.jpg')
    except Exception as e:
        print(f"Processing failed: {e}")
    
    # Example 2: Load image and maintain aspect ratio with max size
    print("\nExample 2: Load image and maintain aspect ratio with max size")
    try:
        # Load image and resize to max dimension 512, maintaining aspect ratio
        image = processor.process_image(
            'sample.jpg',
            output_type='pil',
            max_size=512
        )
        print(f"Resized image dimensions: {image.size}")
        
        # Save as PNG format
        processor.save_image(image, 'output_max512.png', format='PNG')
    except Exception as e:
        print(f"Processing failed: {e}")
    
    # Example 3: Save numpy array as different image formats
    print("\nExample 3: Save numpy array as different image formats")
    try:
        # Create a simple gradient image as numpy array
        width, height = 300, 200
        # Create RGB gradient
        gradient = np.zeros((height, width, 3), dtype=np.uint8)
        for i in range(height):
            for j in range(width):
                gradient[i, j, 0] = int(255 * i / height)      # R channel
                gradient[i, j, 1] = int(255 * j / width)       # G channel
                gradient[i, j, 2] = int(255 * (i+j) / (height+width))  # B channel
        
        print(f"Created numpy array shape: {gradient.shape}")
        
        # Save in different formats
        processor.save_image(gradient, 'numpy_gradient.jpg', format='JPEG', quality=95)
        processor.save_image(gradient, 'numpy_gradient.png', format='PNG')
        processor.save_image(gradient, 'numpy_gradient.webp', format='WEBP')
        
        print("Saved numpy array as JPEG, PNG and WEBP formats")
    except Exception as e:
        print(f"Processing failed: {e}")
    
    # Example 4: Load image from URL and convert to numpy array
    print("\nExample 4: Load image from URL and convert to numpy array")
    try:
        # Use a sample image URL
        url = "https://raw.githubusercontent.com/pytorch/pytorch/master/docs/source/_static/img/pytorch-logo-dark.png"
        
        # Load and convert to numpy array
        numpy_image = processor.process_image(
            url,
            output_type='numpy',
            timeout=15  # Increase timeout
        )
        
        print(f"Image loaded from URL, shape: {numpy_image.shape}")
        
        # Save as JPEG with lower quality to reduce file size
        processor.save_image(numpy_image, 'url_image_lowq.jpg', format='JPEG', quality=50)
        print("Saved as low quality JPEG")
        
        # Save as PNG to maintain quality
        processor.save_image(numpy_image, 'url_image_highq.png', format='PNG')
        print("Saved as high quality PNG")
    except Exception as e:
        print(f"Processing failed: {e}")
    
    # Example 5: Convert image to PyTorch tensor of specific size
    print("\nExample 5: Convert image to PyTorch tensor of specific size")
    try:
        # Load image and convert directly to PyTorch tensor, while resizing
        torch_tensor = processor.process_image(
            'sample.jpg',
            output_type='torch',
            min_size=320,
            max_size=480
        )
        
        print(f"PyTorch tensor shape: {torch_tensor.shape}")
        
        # Convert tensor back to image and save
        processor.save_image(torch_tensor, 'tensor_to_image.jpg')
        print("Saved PyTorch tensor as image")
    except Exception as e:
        print(f"Processing failed: {e}")
    
    # Example 6: Process Base64 encoded image
    print("\nExample 6: Process Base64 encoded image")
    try:
        # First convert image to Base64
        original_image = processor.process_image('sample.jpg')
        base64_str = processor.to_base64(original_image, format='JPEG', quality=90)
        print(f"Generated Base64 string length: {len(base64_str)}")
        
        # Then load image from Base64 and resize
        decoded_image = processor.process_image(
            base64_str,
            output_type='pil',
            min_size=100,
            max_size=800
        )
        
        print(f"Base64 decoded image size: {decoded_image.size}")
        
        # Save in different format
        processor.save_image(decoded_image, 'base64_decoded.webp', format='WEBP')
        print("Saved Base64 decoded image as WEBP format")
    except Exception as e:
        print(f"Processing failed: {e}")
    
    # Example 7: Custom processing parameters
    print("\nExample 7: Using custom processing parameters")
    try:
        # Create a new processor instance with custom default parameters
        custom_processor = ImageConverter(
            min_size=320,
            max_size=1280,
            quality=85,
            convert_to_rgb=True
        )
        
        # Process image
        custom_image = custom_processor.process_image('sample.jpg')
        print(f"Image processed with custom parameters, size: {custom_image.size}")
        
        # Save image
        custom_processor.save_image(custom_image, 'custom_processed.jpg')
        print("Saved custom processed image")
    except Exception as e:
        print(f"Processing failed: {e}")
    
    # Example 8: Exact image resizing to specific resolution
    print("\nExample 8: Exact image resizing to specific resolution")
    try:
        # Load image
        original = processor.process_image('sample.jpg', output_type='pil')
        
        # Manually resize to exact resolution
        exact_size = (640, 480)  # width x height
        resized = original.resize(exact_size, Image.LANCZOS)
        
        print(f"Exactly resized image dimensions: {resized.size}")
        
        # Save resized image
        processor.save_image(resized, 'exact_640x480.jpg')
        print("Saved exactly resized image")
    except Exception as e:
        print(f"Processing failed: {e}")
    
    # Example 9: Batch process multiple images
    print("\nExample 9: Batch process multiple images")
    try:
        # Assume we have a list of images
        image_paths = ['sample1.jpg', 'sample2.jpg', 'sample3.jpg']
        
        # Create output directory
        output_dir = 'processed_images'
        os.makedirs(output_dir, exist_ok=True)
        
        # Batch process
        for i, path in enumerate(image_paths):
            try:
                # Skip non-existent files
                if not os.path.exists(path):
                    print(f"File doesn't exist, skipping: {path}")
                    continue
                
                # Process image
                processed = processor.process_image(
                    path,
                    output_type='pil',
                    max_size=800
                )
                
                # Save processed image
                output_path = os.path.join(output_dir, f"processed_{i}.jpg")
                processor.save_image(processed, output_path)
                print(f"Processed and saved: {output_path}")
            except Exception as e:
                print(f"Processing {path} failed: {e}")
        
        print("Batch processing completed")
    except Exception as e:
        print(f"Batch processing failed: {e}")
    
    # Example 10: Create thumbnail
    print("\nExample 10: Create thumbnail")
    try:
        # Load image
        image = processor.process_image('sample.jpg', output_type='pil')
        
        # Create thumbnail (maintaining aspect ratio)
        thumbnail_size = (150, 150)
        thumbnail = image.copy()
        thumbnail.thumbnail(thumbnail_size, Image.LANCZOS)
        
        print(f"Thumbnail size: {thumbnail.size}")
        
        # Save thumbnail
        processor.save_image(thumbnail, 'thumbnail.jpg', quality=85)
        print("Saved thumbnail")
    except Exception as e:
        print(f"Processing failed: {e}")


if __name__ == "__main__":
    show_examples() 