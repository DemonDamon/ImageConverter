#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Batch Image Processing Operations

This example demonstrates various batch processing techniques:
- Converting multiple images from one format to another
- Batch resizing of images
- Creating image collages and grids
"""

import os
import sys
import glob
from PIL import Image

# Add parent directory to path so we can import the library
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from image_converter import ImageConverter

def batch_convert_images(input_dir, output_dir, target_format='WEBP', resolution=None, quality=90):
    """
    Batch convert all images in a directory to specified format and resolution
    
    Parameters:
        input_dir: Input directory
        output_dir: Output directory
        target_format: Target format, e.g., 'JPEG', 'PNG', 'WEBP', etc.
        resolution: Target resolution, e.g., (800, 600), None to maintain original resolution
        quality: Compression quality
    """
    processor = ImageConverter()
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.tiff']
    image_files = []
    
    for root, _, files in os.walk(input_dir):
        for file in files:
            if any(file.lower().endswith(ext) for ext in image_extensions):
                image_files.append(os.path.join(root, file))
    
    print(f"Found {len(image_files)} image files")
    
    # Process each image
    processed_count = 0
    
    for i, image_path in enumerate(image_files):
        try:
            # Load image
            image = processor.process_image(image_path, output_type='pil')
            
            # Resize if needed
            if resolution is not None:
                image = image.resize(resolution, Image.LANCZOS)
            
            # Build output path
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            output_path = os.path.join(output_dir, f"{base_name}.{target_format.lower()}")
            
            # Save image
            processor.save_image(image, output_path, format=target_format, quality=quality)
            
            processed_count += 1
            print(f"Processed [{i+1}/{len(image_files)}]: {image_path} -> {output_path}")
        except Exception as e:
            print(f"Failed to process {image_path}: {e}")
    
    print(f"Batch conversion completed, processed {processed_count} of {len(image_files)} files")
    return processed_count

def create_image_grid(image_paths, output_path, grid_size=(3, 3), cell_size=(300, 300), format='JPEG'):
    """
    Create an image grid collage
    
    Parameters:
        image_paths: List of image paths
        output_path: Output file path
        grid_size: Grid size, (columns, rows)
        cell_size: Size of each cell, (width, height)
        format: Output format
    """
    processor = ImageConverter()
    
    try:
        # Calculate output image size
        output_width = grid_size[0] * cell_size[0]
        output_height = grid_size[1] * cell_size[1]
        
        # Create blank canvas
        grid_image = Image.new('RGB', (output_width, output_height), color=(255, 255, 255))
        
        # Fill grid
        for i, path in enumerate(image_paths):
            if i >= grid_size[0] * grid_size[1]:
                break  # Exceeds grid size
            
            try:
                # Load image
                img = processor.process_image(path, output_type='pil')
                
                # Resize
                img = img.resize(cell_size, Image.LANCZOS)
                
                # Calculate position
                row = i // grid_size[0]
                col = i % grid_size[0]
                position = (col * cell_size[0], row * cell_size[1])
                
                # Paste into grid
                grid_image.paste(img, position)
                
            except Exception as e:
                print(f"Failed to process {path}: {e}")
                # Create error placeholder
                placeholder = Image.new('RGB', cell_size, color=(200, 200, 200))
                row = i // grid_size[0]
                col = i % grid_size[0]
                position = (col * cell_size[0], row * cell_size[1])
                grid_image.paste(placeholder, position)
        
        # Save grid image
        processor.save_image(grid_image, output_path, format=format)
        print(f"Created image grid: {output_path}")
        
        return output_path
    
    except Exception as e:
        print(f"Failed to create image grid: {e}")
        return None

def batch_resize_with_consistent_naming(input_dir, output_dir, sizes=[(800, 600), (400, 300), (200, 150)]):
    """
    Batch resize images to multiple specified sizes with consistent naming
    
    Parameters:
        input_dir: Input directory containing images
        output_dir: Base output directory
        sizes: List of target sizes (width, height)
    """
    processor = ImageConverter()
    
    # Get all image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp']
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(input_dir, f"*{ext}")))
        image_files.extend(glob.glob(os.path.join(input_dir, f"*{ext.upper()}")))
    
    print(f"Found {len(image_files)} images to resize")
    
    # Process each size
    for size in sizes:
        # Create size-specific output directory
        size_dir = os.path.join(output_dir, f"{size[0]}x{size[1]}")
        os.makedirs(size_dir, exist_ok=True)
        
        print(f"\nResizing to {size[0]}x{size[1]}...")
        
        # Process each image
        for image_path in image_files:
            try:
                # Get base name
                base_name = os.path.splitext(os.path.basename(image_path))[0]
                
                # Load and resize
                image = processor.process_image(image_path, output_type='pil')
                resized = image.resize(size, Image.LANCZOS)
                
                # Determine output format based on original
                orig_ext = os.path.splitext(image_path)[1].lower()
                if orig_ext in ['.jpg', '.jpeg']:
                    out_format = 'JPEG'
                    out_ext = '.jpg'
                elif orig_ext == '.png':
                    out_format = 'PNG'
                    out_ext = '.png'
                else:
                    # Default to JPEG for other formats
                    out_format = 'JPEG'
                    out_ext = '.jpg'
                
                # Save resized image
                output_path = os.path.join(size_dir, f"{base_name}{out_ext}")
                processor.save_image(resized, output_path, format=out_format)
                
                print(f"Resized: {image_path} -> {output_path}")
                
            except Exception as e:
                print(f"Failed to resize {image_path}: {e}")
    
    print("\nBatch resizing completed")
    return output_dir

def main():
    """Demonstrate batch processing operations"""
    # Create output directories
    base_output_dir = os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(base_output_dir, exist_ok=True)
    
    # For testing, you should replace this with your actual image directory
    sample_input_dir = "path/to/image/directory"
    
    print("\n===== Batch Processing Operations =====")
    
    # Example 1: Batch convert to WEBP
    if os.path.exists(sample_input_dir):
        print("\nExample 1: Batch convert images to WEBP format")
        webp_output_dir = os.path.join(base_output_dir, "webp_converted")
        batch_convert_images(
            sample_input_dir,
            webp_output_dir,
            target_format='WEBP',
            quality=85
        )
    
    # Example 2: Create image grid
    # For demo purposes, we'll use glob to find some sample images
    # In real usage, you'd provide specific image paths
    print("\nExample 2: Create image grid collage")
    image_paths = []
    if os.path.exists(sample_input_dir):
        image_paths = glob.glob(os.path.join(sample_input_dir, "*.jpg"))[:9]  # Limit to 9 images
    
    if image_paths:
        grid_output_path = os.path.join(base_output_dir, "image_grid.jpg")
        create_image_grid(
            image_paths,
            grid_output_path,
            grid_size=(3, 3),
            cell_size=(300, 300)
        )
    else:
        print("No sample images found for grid creation")
    
    # Example 3: Batch resize to multiple dimensions
    if os.path.exists(sample_input_dir):
        print("\nExample 3: Batch resize to multiple dimensions")
        resize_output_dir = os.path.join(base_output_dir, "resized")
        batch_resize_with_consistent_naming(
            sample_input_dir,
            resize_output_dir,
            sizes=[(800, 600), (400, 300), (200, 150)]
        )
    
    print("\nBatch processing examples completed")

if __name__ == "__main__":
    main() 