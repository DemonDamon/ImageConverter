#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Advanced Image Processing Examples

This example demonstrates advanced image processing techniques:
- Creating preprocessing pipelines for machine learning models
- Image augmentation for data enhancement
- Advanced transformations and filters
"""

import os
import sys
import random
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

# Add parent directory to path so we can import the library
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from image_converter import ImageConverter

def preprocess_for_model(image_path, target_size=(224, 224), normalize=True):
    """Preprocess an image for machine learning models"""
    processor = ImageConverter()
    
    # Load and resize image
    try:
        image_tensor = processor.process_image(
            image_path,
            output_type='torch',
            min_size=min(target_size),
            max_size=max(target_size)
        )
        
        # Ensure image is the correct size (may need cropping or padding)
        if image_tensor.shape[1:] != target_size:
            # Convert back to PIL for exact resizing
            pil_image = processor.process_image(image_tensor, output_type='pil')
            pil_image = pil_image.resize(target_size, Image.LANCZOS)
            image_tensor = processor.process_image(pil_image, output_type='torch')
        
        # Normalize (if required)
        if normalize:
            try:
                import torch
                # ImageNet normalization
                mean = torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
                std = torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
                image_tensor = (image_tensor - mean) / std
                print(f"Image normalized with ImageNet statistics")
            except ImportError:
                print("PyTorch not available, skipping normalization")
        
        print(f"Image preprocessed to size: {image_tensor.shape}")
        return image_tensor
        
    except Exception as e:
        print(f"Preprocessing failed: {e}")
        return None

def augment_image(image_path, output_dir, num_augmentations=5):
    """Create multiple augmented versions of an image for training"""
    processor = ImageConverter()
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Load original image
        original = processor.process_image(image_path, output_type='pil')
        
        # Save original image
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        processor.save_image(original, os.path.join(output_dir, f"{base_name}_original.jpg"))
        
        print(f"Original image size: {original.size}")
        print(f"Generating {num_augmentations} augmented versions...")
        
        # Create multiple augmented versions
        for i in range(num_augmentations):
            # Copy original image
            img = original.copy()
            
            # Apply various random augmentations
            # 1. Random rotation
            if random.random() > 0.5:
                angle = random.uniform(-15, 15)
                img = img.rotate(angle, Image.BICUBIC, expand=False)
            
            # 2. Random brightness/contrast adjustment
            if random.random() > 0.5:
                factor = random.uniform(0.8, 1.2)
                enhancer = ImageEnhance.Brightness(img)
                img = enhancer.enhance(factor)
            
            if random.random() > 0.5:
                factor = random.uniform(0.8, 1.2)
                enhancer = ImageEnhance.Contrast(img)
                img = enhancer.enhance(factor)
            
            # 3. Random blur
            if random.random() > 0.7:
                img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.5)))
            
            # 4. Random crop and resize
            if random.random() > 0.5:
                width, height = img.size
                crop_percentage = random.uniform(0.8, 0.95)
                crop_width = int(width * crop_percentage)
                crop_height = int(height * crop_percentage)
                left = random.randint(0, width - crop_width)
                top = random.randint(0, height - crop_height)
                img = img.crop((left, top, left + crop_width, top + crop_height))
                img = img.resize((width, height), Image.LANCZOS)
            
            # Save augmented image
            output_path = os.path.join(output_dir, f"{base_name}_aug{i}.jpg")
            processor.save_image(img, output_path)
            print(f"Saved augmented image: {output_path}")
        
        return output_dir
    
    except Exception as e:
        print(f"Image augmentation failed: {e}")
        return None

def save_numpy_with_resolution(numpy_array, output_path, resolution=(1920, 1080), format='JPEG', quality=95):
    """
    Save a numpy array as an image with specific resolution and format
    
    Parameters:
        numpy_array: numpy array, shape (H, W, C) or (H, W)
        output_path: output file path
        resolution: target resolution (width, height)
        format: output format, e.g., 'JPEG', 'PNG', etc.
        quality: JPEG compression quality
    
    Returns:
        Saved file path
    """
    processor = ImageConverter()
    
    try:
        # Ensure numpy array has the correct shape
        if len(numpy_array.shape) == 2:
            # Grayscale image, add channel dimension
            numpy_array = numpy_array[:, :, np.newaxis]
            # Copy to 3 channels
            numpy_array = np.repeat(numpy_array, 3, axis=2)
        
        # If floating point data, convert to uint8
        if numpy_array.dtype == np.float32 or numpy_array.dtype == np.float64:
            # Ensure values are in [0,1] range
            if numpy_array.max() <= 1.0:
                numpy_array = (numpy_array * 255).astype(np.uint8)
            else:
                # Assume values are already in [0,255] range
                numpy_array = numpy_array.astype(np.uint8)
        
        # Convert to PIL image
        pil_image = Image.fromarray(numpy_array)
        
        # Resize to specified resolution
        pil_image = pil_image.resize(resolution, Image.LANCZOS)
        
        # Save image
        result_path = processor.save_image(pil_image, output_path, format=format, quality=quality)
        print(f"Numpy array saved as image with resolution {resolution}: {result_path}")
        return result_path
    
    except Exception as e:
        print(f"Failed to save numpy array: {e}")
        return None

def main():
    """Demonstrate advanced image processing examples"""
    # Create output directory
    output_dir = os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(output_dir, exist_ok=True)
    
    # Replace with your actual image path
    sample_image_path = "path/to/sample_image.jpg"
    
    print("\n===== Advanced Image Processing Examples =====")
    
    # Example 1: Preprocess for model
    if os.path.exists(sample_image_path):
        print("\nExample 1: Preprocess image for ML model")
        preprocessed_tensor = preprocess_for_model(
            sample_image_path, 
            target_size=(224, 224), 
            normalize=True
        )
        
        if preprocessed_tensor is not None:
            print(f"Successfully preprocessed image to tensor of shape: {preprocessed_tensor.shape}")
    
    # Example 2: Image augmentation
    if os.path.exists(sample_image_path):
        print("\nExample 2: Image augmentation")
        augmentation_dir = os.path.join(output_dir, "augmentations")
        augment_image(sample_image_path, augmentation_dir, num_augmentations=3)
    
    # Example 3: Create and save a numpy array with specific resolution
    print("\nExample 3: Creating and saving numpy array with specific resolution")
    # Create a simple gradient as numpy array
    width, height = 300, 200
    gradient = np.zeros((height, width, 3), dtype=np.float32)
    for i in range(height):
        for j in range(width):
            gradient[i, j, 0] = i / height  # R channel
            gradient[i, j, 1] = j / width   # G channel
            gradient[i, j, 2] = (i+j) / (height+width)  # B channel
    
    # Save with specific resolution
    output_path = os.path.join(output_dir, "custom_resolution.jpg")
    save_numpy_with_resolution(
        gradient, 
        output_path, 
        resolution=(1280, 720), 
        format='JPEG', 
        quality=95
    )
    
    print("\nAdvanced processing examples completed")

if __name__ == "__main__":
    main() 