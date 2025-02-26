#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Web Images Processing Examples

This example demonstrates how to:
- Load images from URLs
- Process images from web sources
- Handle web image errors and timeouts
"""

import os
import sys
import time

# Add parent directory to path so we can import the library
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from image_converter import ImageConverter

def main():
    """Demonstrate web image processing functionality"""
    # Initialize the image processor
    processor = ImageConverter()
    
    # Create output directory for saving images
    output_dir = os.path.join(os.path.dirname(__file__), "output")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Example URLs - replace with valid image URLs for testing
    # Using placeholder URLs that need to be replaced for actual testing
    image_urls = [
        "https://example.com/sample1.jpg",  # Replace with real URL
        "https://example.com/sample2.png",  # Replace with real URL
        "https://example.com/non_existent.jpg",  # Intentionally invalid URL for error handling demo
    ]
    
    # Example 1: Basic URL loading
    print("\nExample 1: Basic URL loading")
    for i, url in enumerate(image_urls):
        try:
            print(f"Processing URL: {url}")
            start_time = time.time()
            
            # Process the image with default parameters
            image = processor.process_image(url)
            
            elapsed_time = time.time() - start_time
            print(f"  Success! Image size: {image.size}, loaded in {elapsed_time:.2f} seconds")
            
            # Save the image
            output_path = os.path.join(output_dir, f"url_image_{i}.jpg")
            processor.save_image(image, output_path)
            print(f"  Saved to: {output_path}")
            
        except Exception as e:
            print(f"  Error processing URL: {e}")
    
    # Example 2: URL loading with custom timeout and size constraints
    print("\nExample 2: URL loading with custom parameters")
    try:
        # Use the first URL that worked in Example 1
        for url in image_urls:
            try:
                print(f"Processing URL with custom parameters: {url}")
                
                # Custom parameters
                image = processor.process_image(
                    url,
                    timeout=5,  # 5 second timeout
                    min_size=300,  # Minimum dimension 300px
                    max_size=800,  # Maximum dimension 800px
                    convert_to_rgb=True
                )
                
                print(f"  Success! Image size after resizing: {image.size}")
                
                # Save the image
                output_path = os.path.join(output_dir, "url_image_custom.jpg")
                processor.save_image(image, output_path)
                print(f"  Saved to: {output_path}")
                
                # Once we've processed one URL successfully, break the loop
                break
                
            except Exception:
                # Try the next URL
                continue
    except Exception as e:
        print(f"  Error with custom parameters: {e}")
    
    # Example 3: Converting web image to different formats
    print("\nExample 3: Converting web image to different formats")
    try:
        # Try the first URL
        url = image_urls[0]
        
        # Convert to numpy array
        numpy_array = processor.process_image(url, output_type='numpy')
        print(f"Converted to numpy array, shape: {numpy_array.shape}")
        
        # Convert to torch tensor if available
        try:
            torch_tensor = processor.process_image(url, output_type='torch')
            print(f"Converted to PyTorch tensor, shape: {torch_tensor.shape}")
        except ImportError:
            print("PyTorch not available, skipping tensor conversion")
        
    except Exception as e:
        print(f"Error converting web image: {e}")
    
    print("\nWeb image examples completed")

if __name__ == "__main__":
    main() 