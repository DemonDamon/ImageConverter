#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Save and Convert Examples

This example demonstrates how to:
- Save processed images to different file formats
- Convert images to Base64 strings
- Handle different input and output formats
"""

import os
import sys
import numpy as np

# Add parent directory to path so we can import the library
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from image_converter import ImageConverter

def main():
    """Demonstrate save and convert functionality"""
    # Initialize the image processor
    processor = ImageConverter()
    
    # You should replace this with an actual image path for testing
    sample_image_path = "path/to/sample_image.jpg"
    
    # Create output directory if it doesn't exist
    output_dir = os.path.join(os.path.dirname(__file__), "output")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Example 1: Save image to different formats
    try:
        if os.path.exists(sample_image_path):
            # Load the image
            pil_image = processor.process_image(sample_image_path)
            
            # Save as JPEG
            jpeg_path = os.path.join(output_dir, "output_jpeg.jpg")
            processor.save_image(pil_image, jpeg_path, format='JPEG', quality=90)
            print(f"Image saved as JPEG: {jpeg_path}")
            
            # Save as PNG
            png_path = os.path.join(output_dir, "output_png.png")
            processor.save_image(pil_image, png_path, format='PNG')
            print(f"Image saved as PNG: {png_path}")
            
            # Save as WEBP
            webp_path = os.path.join(output_dir, "output_webp.webp")
            processor.save_image(pil_image, webp_path, format='WEBP')
            print(f"Image saved as WEBP: {webp_path}")
    except Exception as e:
        print(f"Error saving images: {e}")
    
    # Example 2: Convert to Base64
    try:
        if os.path.exists(sample_image_path):
            # Load the image
            pil_image = processor.process_image(sample_image_path)
            
            # Convert to Base64 with data URI
            base64_with_uri = processor.to_base64(
                pil_image, 
                format='JPEG', 
                quality=90,
                include_data_uri=True
            )
            print(f"Base64 with URI prefix length: {len(base64_with_uri)}")
            print(f"Base64 starts with: {base64_with_uri[:50]}...")
            
            # Convert to Base64 without data URI
            base64_without_uri = processor.to_base64(
                pil_image, 
                format='JPEG', 
                quality=90,
                include_data_uri=False
            )
            print(f"Base64 without URI prefix length: {len(base64_without_uri)}")
    except Exception as e:
        print(f"Error converting to Base64: {e}")
    
    # Example 3: Save numpy array as image
    try:
        if os.path.exists(sample_image_path):
            # Load image as numpy array
            numpy_image = processor.process_image(sample_image_path, output_type='numpy')
            
            # Save numpy array as image
            numpy_output_path = os.path.join(output_dir, "numpy_output.jpg")
            processor.save_image(numpy_image, numpy_output_path)
            print(f"Numpy array saved as image: {numpy_output_path}")
    except Exception as e:
        print(f"Error saving numpy array: {e}")
    
    # Example 4: Save PyTorch tensor as image (if available)
    try:
        if os.path.exists(sample_image_path):
            try:
                # Try to load as torch tensor
                torch_image = processor.process_image(sample_image_path, output_type='torch')
                
                # Save tensor as image
                tensor_output_path = os.path.join(output_dir, "tensor_output.png")
                processor.save_image(torch_image, tensor_output_path, format='PNG')
                print(f"PyTorch tensor saved as image: {tensor_output_path}")
            except ImportError:
                print("PyTorch is not installed, skipping tensor example")
    except Exception as e:
        print(f"Error saving PyTorch tensor: {e}")
    
    print("Save and convert examples completed")

if __name__ == "__main__":
    main() 