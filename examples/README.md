# ImageConverter Examples

This directory contains example scripts demonstrating how to use the ImageConverter library for various image processing tasks.

## Prerequisites

Before running these examples, make sure you have installed the required dependencies:

```bash
pip install pillow numpy requests
# Optional dependencies for extended functionality
pip install torch
pip install pillow_heif  # For HEIF/HEIC support
```

## Example Files

### 1. Basic Usage (`basic_usage.py`)

Demonstrates the fundamental usage of the ImageConverter library:
- Loading images from different sources
- Converting between different output formats
- Basic image processing operations

```bash
python basic_usage.py
```

### 2. Save and Convert (`save_and_convert.py`)

Shows how to:
- Save processed images in different formats
- Convert images between PIL, NumPy arrays, and PyTorch tensors
- Adjust quality settings

```bash
python save_and_convert.py
```

### 3. Web Images (`web_images.py`)

Examples for handling images from web sources:
- Loading images from URLs
- Processing remote images
- Error handling for web requests

```bash
python web_images.py
```

### 4. Base64 Processing (`base64_processing.py`)

Demonstrates working with Base64-encoded images:
- Converting images to/from Base64 strings
- Processing Base64-encoded images
- Handling different Base64 formats

```bash
python base64_processing.py
```

### 5. Common Examples (`common_examples.py`)

A comprehensive collection of common use cases:
- Resizing images to specific dimensions
- Creating gradient images with numpy
- Processing URL images
- Creating thumbnails
- Batch processing multiple images

```bash
python common_examples.py
```

### 6. Advanced Processing (`advanced_processing.py`)

Advanced image processing techniques for specialized use cases:
- Creating preprocessing pipelines for machine learning models
- Image augmentation for data enhancement
- Handling numpy arrays with specific resolutions
- Normalizing images for model inference

```bash
python advanced_processing.py
```

### 7. Batch Operations (`batch_operations.py`)

Batch processing operations for working with multiple images:
- Converting all images in a directory to another format
- Creating image grid collages
- Batch resizing to multiple dimensions with consistent naming

```bash
python batch_operations.py
```

### 8. Special Effects (`special_effects.py`)

Creative and artistic image processing effects:
- Creating image grid collages from multiple images
- Applying artistic filters (sketch, watercolor, vintage, negative)
- Creating gradient and pattern overlays
- Generating thumbnail galleries

```bash
python special_effects.py
```

## Output Directory

Each example script creates an `output` directory to store generated files. This includes:
- Processed images
- Converted formats
- Base64 text files
- Image grids and collages
- Augmented images

## Customization

To run these examples with your own images:
1. Replace the placeholder file paths in each script with your actual image paths
2. Replace example URLs with valid image URLs
3. Adjust parameters like `min_size`, `max_size`, and `quality` as needed

## Notes

- Some examples require PyTorch to be installed and will skip those sections if not available
- Most examples have error handling to demonstrate robust usage patterns
- The advanced examples assume familiarity with numpy and image processing concepts 