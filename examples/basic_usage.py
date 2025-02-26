#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Basic Usage Examples of ImageConverter

This example demonstrates the basic functionality of the ImageConverter:
- Loading images from different sources (file, URL, base64)
- Converting to different output formats (PIL, numpy, torch)
- Basic image processing
"""

import os
import sys
import numpy as np

# Add parent directory to path so we can import the library
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from image_converter import ImageConverter

def main():
    """Demonstrate basic usage of ImageConverter"""
    # Initialize the image processor with default parameters
    processor = ImageConverter()
    print("ImageConverter initialized with default parameters")
    
    # You should replace these paths with actual images for testing
    sample_image_path = "path/to/sample_image.jpg"
    sample_url = "https://example.com/sample_image.jpg"
    
    # Example 1: Process a local file and convert to different formats
    try:
        # If you don't have a local image, you can comment this section out
        if os.path.exists(sample_image_path):
            # Convert to PIL Image (default)
            pil_image = processor.process_image(sample_image_path)
            print(f"PIL Image size: {pil_image.size}")
            
            # Convert to numpy array
            numpy_image = processor.process_image(sample_image_path, output_type='numpy')
            print(f"Numpy array shape: {numpy_image.shape}")
            
            # Convert to PyTorch tensor if PyTorch is available
            try:
                torch_image = processor.process_image(sample_image_path, output_type='torch')
                print(f"PyTorch tensor shape: {torch_image.shape}")
            except ImportError:
                print("PyTorch is not installed, skipping tensor conversion")
    except Exception as e:
        print(f"Error processing local file: {e}")
    
    # Example 2: Process an image from URL
    try:
        # This will fail if the URL doesn't exist or doesn't point to an image
        # Replace with a valid image URL for testing
        pil_image = processor.process_image(
            sample_url,
            timeout=5  # Custom timeout parameter
        )
        print(f"Successfully loaded image from URL, size: {pil_image.size}")
    except Exception as e:
        print(f"Error loading from URL: {e}")
    
    # Example 3: Custom processing parameters
    try:
        if os.path.exists(sample_image_path):
            # Specify custom processing parameters
            pil_image = processor.process_image(
                sample_image_path,
                min_size=224,  # Ensure image is at least 224px on smallest side
                max_size=1024,  # Ensure image is at most 1024px on largest side
                convert_to_rgb=True  # Ensure image is in RGB mode
            )
            print(f"Processed with custom parameters, new size: {pil_image.size}")
    except Exception as e:
        print(f"Error with custom processing: {e}")
    
    print("Basic usage examples completed")

if __name__ == "__main__":
    main() 