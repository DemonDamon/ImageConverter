#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Special Effects and Image Collage Examples

This script demonstrates advanced image processing techniques using the ImageConverter library, 
including creating image collages and applying special effects. The examples show how to:

1. Create image grid collages from multiple images
2. Apply artistic filters and effects to images
3. Create gradient and pattern overlays
4. Generate thumbnail galleries
5. Combine multiple images with blending modes

All examples save their output to the "output" directory.
"""

import os
import sys
import random
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

# Add parent directory to path to import ImageConverter
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from image_converter import ImageConverter


def create_image_grid(image_paths, output_path, grid_size=(3, 3), cell_size=(300, 300), format='JPEG'):
    """
    Create an image grid collage from multiple images
    
    Args:
        image_paths: List of image file paths
        output_path: Output file path
        grid_size: Grid dimensions as (columns, rows)
        cell_size: Size of each cell as (width, height)
        format: Output image format
    
    Returns:
        Path to the saved grid image
    """
    processor = ImageConverter()
    
    # Calculate output image dimensions
    output_width = grid_size[0] * cell_size[0]
    output_height = grid_size[1] * cell_size[1]
    
    # Create a blank canvas
    grid_image = Image.new('RGB', (output_width, output_height), color=(255, 255, 255))
    
    # Fill the grid
    for i, path in enumerate(image_paths):
        if i >= grid_size[0] * grid_size[1]:
            break  # Exceeded grid capacity
        
        try:
            # Load the image
            img = processor.process_image(path, output_type='pil')
            
            # Resize to fit the cell
            img = img.resize(cell_size, Image.LANCZOS)
            
            # Calculate position
            row = i // grid_size[0]
            col = i % grid_size[0]
            position = (col * cell_size[0], row * cell_size[1])
            
            # Paste into the grid
            grid_image.paste(img, position)
            
        except Exception as e:
            print(f"Failed to process {path}: {e}")
            # Create a placeholder for errors
            placeholder = Image.new('RGB', cell_size, color=(200, 200, 200))
            row = i // grid_size[0]
            col = i % grid_size[0]
            position = (col * cell_size[0], row * cell_size[1])
            grid_image.paste(placeholder, position)
    
    # Save the grid image
    processor.save_image(grid_image, output_path, format=format)
    print(f"Created image grid: {output_path}")
    
    return output_path


def apply_artistic_filter(image_path, output_path, filter_type='sketch'):
    """
    Apply artistic filters to an image
    
    Args:
        image_path: Input image path
        output_path: Output file path
        filter_type: Type of filter to apply ('sketch', 'watercolor', 'vintage', 'negative')
    
    Returns:
        Path to the saved filtered image
    """
    processor = ImageConverter()
    
    # Load the image
    img = processor.process_image(image_path, output_type='pil')
    
    # Apply the selected filter
    if filter_type == 'sketch':
        # Create a sketch effect
        img_gray = img.convert('L')
        img_inverted = ImageOps.invert(img_gray)
        img_blurred = img_inverted.filter(ImageFilter.GaussianBlur(radius=10))
        img_sketch = Image.blend(img_gray, img_blurred, alpha=0.5)
        result = img_sketch
        
    elif filter_type == 'watercolor':
        # Create a watercolor-like effect
        img_blurred = img.filter(ImageFilter.GaussianBlur(radius=2))
        enhancer = ImageEnhance.Color(img_blurred)
        img_saturated = enhancer.enhance(1.5)
        enhancer = ImageEnhance.Contrast(img_saturated)
        result = enhancer.enhance(1.2)
        
    elif filter_type == 'vintage':
        # Create a vintage/sepia effect
        img_sepia = img.convert('L')
        img_sepia = ImageOps.colorize(img_sepia, "#704214", "#C0A080")
        enhancer = ImageEnhance.Contrast(img_sepia)
        result = enhancer.enhance(0.8)
        
    elif filter_type == 'negative':
        # Create a negative image
        result = ImageOps.invert(img)
        
    else:
        # Default: return original
        result = img
    
    # Save the filtered image
    processor.save_image(result, output_path)
    print(f"Applied {filter_type} filter: {output_path}")
    
    return output_path


def create_gradient_overlay(image_path, output_path, gradient_direction='horizontal', opacity=0.5):
    """
    Create a gradient overlay on an image
    
    Args:
        image_path: Input image path
        output_path: Output file path
        gradient_direction: Direction of gradient ('horizontal', 'vertical', 'diagonal')
        opacity: Opacity of the gradient overlay (0.0 to 1.0)
    
    Returns:
        Path to the saved image with gradient overlay
    """
    processor = ImageConverter()
    
    # Load the image
    img = processor.process_image(image_path, output_type='pil')
    width, height = img.size
    
    # Create gradient overlay
    gradient = Image.new('RGBA', (width, height), (0, 0, 0, 0))
    
    # Generate gradient data
    gradient_data = []
    for y in range(height):
        for x in range(width):
            if gradient_direction == 'horizontal':
                alpha = int(255 * x / width)
            elif gradient_direction == 'vertical':
                alpha = int(255 * y / height)
            elif gradient_direction == 'diagonal':
                alpha = int(255 * (x + y) / (width + height))
            else:
                alpha = 128
            
            gradient_data.append((255, 100, 100, alpha))
    
    # Apply gradient data
    gradient.putdata(gradient_data)
    
    # Convert original image to RGBA if not already
    if img.mode != 'RGBA':
        img = img.convert('RGBA')
    
    # Blend images
    result = Image.alpha_composite(img, Image.blend(Image.new('RGBA', img.size, (0, 0, 0, 0)), gradient, opacity))
    
    # Save the result
    processor.save_image(result, output_path)
    print(f"Created gradient overlay: {output_path}")
    
    return output_path


def create_thumbnail_gallery(image_dir, output_path, thumbnail_size=(150, 150), padding=10, max_cols=5):
    """
    Create a thumbnail gallery from images in a directory
    
    Args:
        image_dir: Directory containing images
        output_path: Output file path
        thumbnail_size: Size of each thumbnail
        padding: Padding between thumbnails
        max_cols: Maximum number of columns
    
    Returns:
        Path to the saved gallery image
    """
    processor = ImageConverter()
    
    # Find all images in the directory
    image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp']
    image_files = []
    
    for file in os.listdir(image_dir):
        if any(file.lower().endswith(ext) for ext in image_extensions):
            image_files.append(os.path.join(image_dir, file))
    
    if not image_files:
        print(f"No images found in {image_dir}")
        return None
    
    # Calculate gallery dimensions
    num_images = len(image_files)
    num_cols = min(max_cols, num_images)
    num_rows = (num_images + num_cols - 1) // num_cols  # Ceiling division
    
    gallery_width = num_cols * (thumbnail_size[0] + padding) + padding
    gallery_height = num_rows * (thumbnail_size[1] + padding) + padding
    
    # Create blank gallery image
    gallery = Image.new('RGB', (gallery_width, gallery_height), color=(240, 240, 240))
    
    # Place thumbnails
    for i, img_path in enumerate(image_files):
        try:
            # Load and resize the image
            img = processor.process_image(img_path, output_type='pil')
            thumb = img.copy()
            thumb.thumbnail(thumbnail_size, Image.LANCZOS)
            
            # Calculate position
            row = i // num_cols
            col = i % num_cols
            pos_x = padding + col * (thumbnail_size[0] + padding)
            pos_y = padding + row * (thumbnail_size[1] + padding)
            
            # Center the thumbnail if smaller than thumbnail_size
            offset_x = (thumbnail_size[0] - thumb.width) // 2
            offset_y = (thumbnail_size[1] - thumb.height) // 2
            
            gallery.paste(thumb, (pos_x + offset_x, pos_y + offset_y))
            
        except Exception as e:
            print(f"Failed to process {img_path}: {e}")
    
    # Save the gallery
    processor.save_image(gallery, output_path)
    print(f"Created thumbnail gallery: {output_path}")
    
    return output_path


def main():
    """Main function to demonstrate special effects examples"""
    # Create output directory
    output_dir = os.path.join(os.path.dirname(__file__), 'output')
    os.makedirs(output_dir, exist_ok=True)
    
    # Sample image paths (replace with actual paths)
    sample_image = "path/to/sample_image.jpg"  # Replace with an actual image path
    sample_dir = "path/to/image_directory"     # Replace with an actual directory path
    
    # Example image paths for grid (replace with actual paths)
    image_paths = [
        "path/to/image1.jpg",
        "path/to/image2.jpg",
        "path/to/image3.jpg",
        "path/to/image4.jpg",
        "path/to/image5.jpg",
        "path/to/image6.jpg",
    ]
    
    # Example 1: Create an image grid
    print("\nExample 1: Creating an image grid collage")
    try:
        grid_output = os.path.join(output_dir, "image_grid_collage.jpg")
        create_image_grid(
            image_paths,
            grid_output,
            grid_size=(3, 2),
            cell_size=(320, 240)
        )
    except Exception as e:
        print(f"Failed to create image grid: {e}")
    
    # Example 2: Apply artistic filters
    print("\nExample 2: Applying artistic filters")
    try:
        # Apply multiple filters to the same image
        if os.path.exists(sample_image):
            for filter_type in ['sketch', 'watercolor', 'vintage', 'negative']:
                filter_output = os.path.join(output_dir, f"{filter_type}_effect.jpg")
                apply_artistic_filter(
                    sample_image,
                    filter_output,
                    filter_type=filter_type
                )
    except Exception as e:
        print(f"Failed to apply artistic filters: {e}")
    
    # Example 3: Create gradient overlays
    print("\nExample 3: Creating gradient overlays")
    try:
        if os.path.exists(sample_image):
            for direction in ['horizontal', 'vertical', 'diagonal']:
                overlay_output = os.path.join(output_dir, f"{direction}_gradient_overlay.png")
                create_gradient_overlay(
                    sample_image,
                    overlay_output,
                    gradient_direction=direction,
                    opacity=0.6
                )
    except Exception as e:
        print(f"Failed to create gradient overlays: {e}")
    
    # Example 4: Create a thumbnail gallery
    print("\nExample 4: Creating a thumbnail gallery")
    try:
        if os.path.exists(sample_dir):
            gallery_output = os.path.join(output_dir, "thumbnail_gallery.jpg")
            create_thumbnail_gallery(
                sample_dir,
                gallery_output,
                thumbnail_size=(150, 150),
                padding=10,
                max_cols=4
            )
    except Exception as e:
        print(f"Failed to create thumbnail gallery: {e}")
    
    print("\nSpecial effects examples completed. Outputs saved to:", output_dir)


if __name__ == "__main__":
    main() 