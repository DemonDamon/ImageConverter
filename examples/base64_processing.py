#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Base64 Image Processing Examples

This example demonstrates how to:
- Encode images to Base64 strings
- Decode Base64 strings to images
- Process Base64 encoded images
"""

import os
import sys
import base64
from PIL import Image
import io

# Add parent directory to path so we can import the library
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from image_converter import ImageConverter

def main():
    """Demonstrate Base64 image processing functionality"""
    # Initialize the image processor
    processor = ImageConverter()
    
    # Create output directory for saving images
    output_dir = os.path.join(os.path.dirname(__file__), "output")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # You should replace this with an actual image path for testing
    sample_image_path = "path/to/sample_image.jpg"
    
    # Example 1: Convert an image to Base64
    try:
        print("\nExample 1: Convert image to Base64")
        if os.path.exists(sample_image_path):
            # Load the image
            pil_image = processor.process_image(sample_image_path)
            print(f"Original image size: {pil_image.size}")
            
            # Convert to Base64 with data URI
            base64_str = processor.to_base64(pil_image, format='JPEG', quality=90)
            print(f"Generated Base64 string length: {len(base64_str)}")
            print(f"Base64 string starts with: {base64_str[:50]}...")
            
            # Write Base64 string to file for reference
            base64_file = os.path.join(output_dir, "image_base64.txt")
            with open(base64_file, "w") as f:
                f.write(base64_str)
            print(f"Base64 string saved to: {base64_file}")
    except Exception as e:
        print(f"Error converting to Base64: {e}")
    
    # Example 2: Process a Base64 encoded image
    try:
        print("\nExample 2: Process Base64 encoded image")
        if os.path.exists(sample_image_path):
            # First generate a Base64 string from a local image
            with open(sample_image_path, "rb") as img_file:
                img_bytes = img_file.read()
                base64_str = "data:image/jpeg;base64," + base64.b64encode(img_bytes).decode('utf-8')
            
            print(f"Generated Base64 string length: {len(base64_str)}")
            
            # Process the Base64 string
            processed_image = processor.process_image(
                base64_str,
                min_size=300,
                max_size=800
            )
            print(f"Processed Base64 image size: {processed_image.size}")
            
            # Save the processed image
            output_path = os.path.join(output_dir, "processed_from_base64.jpg")
            processor.save_image(processed_image, output_path)
            print(f"Processed image saved to: {output_path}")
    except Exception as e:
        print(f"Error processing Base64 image: {e}")
    
    # Example 3: Handle different Base64 formats
    print("\nExample 3: Handle different Base64 formats")
    
    # Test cases for different Base64 formats
    if os.path.exists(sample_image_path):
        try:
            # Standard Base64 with data URI
            with open(sample_image_path, "rb") as img_file:
                img_bytes = img_file.read()
                standard_base64 = "data:image/jpeg;base64," + base64.b64encode(img_bytes).decode('utf-8')
            
            # Base64 without data URI
            raw_base64 = base64.b64encode(img_bytes).decode('utf-8')
            
            # URL-safe Base64 (replace + with - and / with _)
            url_safe_base64 = standard_base64.replace("+", "-").replace("/", "_")
            
            # Process each format
            formats = {
                "Standard Base64 with URI": standard_base64,
                "Raw Base64 without URI": raw_base64,
                "URL-safe Base64": url_safe_base64
            }
            
            for name, b64_str in formats.items():
                try:
                    img = processor.process_image(b64_str)
                    print(f"Successfully processed {name}, image size: {img.size}")
                except Exception as e:
                    print(f"Error processing {name}: {e}")
        
        except Exception as e:
            print(f"Error in Base64 format examples: {e}")
    
    # Example 4: Convert between different formats via Base64
    print("\nExample 4: Convert between formats via Base64")
    try:
        if os.path.exists(sample_image_path):
            # Load image and convert to Base64
            original = processor.process_image(sample_image_path)
            base64_str = processor.to_base64(original, format='JPEG')
            
            # Process the Base64 string to different output formats
            pil_from_base64 = processor.process_image(base64_str, output_type='pil')
            numpy_from_base64 = processor.process_image(base64_str, output_type='numpy')
            
            print(f"Converted Base64 back to PIL image: {pil_from_base64.size}")
            print(f"Converted Base64 to numpy array: {numpy_from_base64.shape}")
            
            # Try converting to torch if available
            try:
                torch_from_base64 = processor.process_image(base64_str, output_type='torch')
                print(f"Converted Base64 to PyTorch tensor: {torch_from_base64.shape}")
            except ImportError:
                print("PyTorch not available, skipping tensor conversion")
    except Exception as e:
        print(f"Error converting between formats: {e}")
    
    print("\nBase64 examples completed")

if __name__ == "__main__":
    main() 