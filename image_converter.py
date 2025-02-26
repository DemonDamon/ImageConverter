#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import io
import re
import base64
import logging
import math
from typing import Union, Optional, Tuple, List, Dict, Any
from urllib.parse import urlparse
import traceback

import numpy as np
import requests
from PIL import Image, UnidentifiedImageError

# Optional dependencies, skip related features if not present
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
    HAS_HEIF_SUPPORT = True
except ImportError:
    HAS_HEIF_SUPPORT = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("image_converter")

# Remove PIL's maximum image pixel limit
Image.MAX_IMAGE_PIXELS = None


class ImageConverter:
    """Universal image converter that supports multiple input sources and output formats"""
    
    # Supported image formats and corresponding MIME types
    MIME_TYPES = {
        '.jpg': 'jpeg',
        '.jpeg': 'jpeg',
        '.png': 'png',
        '.gif': 'gif',
        '.webp': 'webp',
        '.bmp': 'bmp',
        '.heic': 'heic',
        '.heif': 'heif',
    }
    
    # Default image processing parameters
    DEFAULT_PARAMS = {
        'min_size': 56,       # Minimum size (short edge)
        'max_size': 3584,     # Maximum size (long edge)
        'quality': 95,        # JPEG compression quality
        'timeout': 10,        # URL request timeout (seconds)
        'convert_to_rgb': True,  # Whether to convert to RGB mode
    }
    
    def __init__(self, **kwargs):
        """
        Initialize the image converter
        
        Parameters:
            min_size (int): Minimum image size (short edge), default 56
            max_size (int): Maximum image size (long edge), default 3584
            quality (int): JPEG compression quality, default 95
            timeout (int): URL request timeout (seconds), default 10
            convert_to_rgb (bool): Whether to convert to RGB mode, default True
        """
        self.params = self.DEFAULT_PARAMS.copy()
        self.params.update(kwargs)
        logger.info(f"Initializing image converter, parameters: {self.params}")
    
    def process_image(
        self, 
        image_source: Union[str, bytes, Image.Image],
        output_type: str = 'pil',
        **kwargs
    ) -> Union[Image.Image, np.ndarray, 'torch.Tensor']:
        """
        Process image and return specified format
        
        Parameters:
            image_source: Image source, can be file path, URL, Base64 string, PIL image object
            output_type: Output type, 'pil', 'numpy' or 'torch'
            **kwargs: Other processing parameters, will override default parameters
            
        Returns:
            Image data in the corresponding format according to output_type
        """
        # Update processing parameters
        process_params = self.params.copy()
        process_params.update(kwargs)
        
        try:
            # 1. Load image
            pil_image = self._load_image(image_source, process_params)
            
            # 2. Preprocess image
            pil_image = self._preprocess_image(pil_image, process_params)
            
            # 3. Convert to requested output format
            return self._convert_to_output_format(pil_image, output_type)
            
        except Exception as e:
            logger.error(f"Image processing failed: {str(e)}")
            logger.debug(traceback.format_exc())
            raise
    
    def _load_image(self, image_source: Union[str, bytes, Image.Image], params: Dict[str, Any]) -> Image.Image:
        """
        Load image from different sources
        
        Parameters:
            image_source: Image source
            params: Processing parameters
            
        Returns:
            PIL image object
        """
        # If already a PIL image, return directly
        if isinstance(image_source, Image.Image):
            logger.info("Input is already a PIL image object")
            return image_source
        
        # If byte data, decode directly
        if isinstance(image_source, bytes):
            logger.info("Loading image from byte data")
            return self._load_image_from_bytes(image_source)
        
        # Handle string type input
        if isinstance(image_source, str):
            # Check if Base64 encoded
            if self._is_base64(image_source):
                logger.info("Loading image from Base64 string")
                return self._load_image_from_base64(image_source)
            
            # Check if URL
            if self._is_url(image_source):
                logger.info(f"Loading image from URL: {image_source}")
                return self._load_image_from_url(image_source, params['timeout'])
            
            # Check if local file path
            if os.path.exists(image_source):
                logger.info(f"Loading image from local file: {image_source}")
                return self._load_image_from_file(image_source)
            
            # Assume it's a storage path, try to parse
            logger.info(f"Attempting to load image from storage path: {image_source}")
            return self._load_image_from_storage(image_source)
        
        # Unsupported input type
        raise ValueError(f"Unsupported image source type: {type(image_source)}")
    
    def _is_base64(self, s: str) -> bool:
        """Determine if string is Base64 encoded"""
        # Check if has Base64 prefix
        if s.startswith('data:image/'):
            return True
        
        # Check if matches Base64 encoding rules
        pattern = r'^[A-Za-z0-9+/]+={0,2}$'
        return bool(re.match(pattern, s))
    
    def _is_url(self, s: str) -> bool:
        """Determine if string is a URL"""
        try:
            result = urlparse(s)
            return all([result.scheme, result.netloc])
        except:
            return False
    
    def _load_image_from_bytes(self, image_bytes: bytes) -> Image.Image:
        """Load image from byte data"""
        try:
            return Image.open(io.BytesIO(image_bytes))
        except UnidentifiedImageError:
            raise ValueError("Unrecognized image format")
        except Exception as e:
            raise ValueError(f"Failed to load image from byte data: {str(e)}")
    
    def _load_image_from_base64(self, base64_str: str) -> Image.Image:
        """Load image from Base64 string"""
        try:
            # Remove data URI prefix
            if base64_str.startswith('data:image/'):
                base64_str = base64_str.split(',', 1)[1]
            
            # Check if URL-safe base64 encoding
            if not set(base64_str).issubset(set('ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/=')):
                # Might be URL-safe base64, try to convert
                base64_str = base64_str.replace('-', '+').replace('_', '/')
            
            # Add missing padding
            padding = 4 - (len(base64_str) % 4) if len(base64_str) % 4 else 0
            base64_str = base64_str + ('=' * padding)
            
            # Decode
            image_bytes = base64.b64decode(base64_str)
            return self._load_image_from_bytes(image_bytes)
        except Exception as e:
            raise ValueError(f"Base64 decoding failed: {str(e)}")
    
    def _load_image_from_url(self, url: str, timeout: int) -> Image.Image:
        """Load image from URL"""
        try:
            response = requests.get(url, timeout=timeout, stream=True)
            response.raise_for_status()
            
            # Check content type
            content_type = response.headers.get('Content-Type', '')
            if not content_type.startswith('image/'):
                raise ValueError(f"URL did not return image content: {content_type}")
            
            return self._load_image_from_bytes(response.content)
        except requests.RequestException as e:
            raise ValueError(f"URL request failed: {str(e)}")
    
    def _load_image_from_file(self, file_path: str) -> Image.Image:
        """Load image from local file"""
        try:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File does not exist: {file_path}")
            
            return Image.open(file_path)
        except UnidentifiedImageError:
            raise ValueError(f"Unrecognized image format: {file_path}")
        except Exception as e:
            raise ValueError(f"Failed to load file: {str(e)}")
    
    def _load_image_from_storage(self, storage_path: str) -> Image.Image:
        """Load image from storage system (extensible implementation for different storage systems)"""
        # Implement different storage systems as needed, such as S3, OSS, etc.
        # Default attempt to treat as local path
        try:
            return self._load_image_from_file(storage_path)
        except FileNotFoundError:
            raise ValueError(f"Unable to load image from storage path: {storage_path}")
    
    def _preprocess_image(self, image: Image.Image, params: Dict[str, Any]) -> Image.Image:
        """
        Preprocess image: resize, convert mode, etc.
        
        Parameters:
            image: PIL image object
            params: Processing parameters
            
        Returns:
            Processed PIL image object
        """
        # Record original mode
        original_mode = image.mode
        
        # Convert to RGB mode (if needed)
        if params['convert_to_rgb'] and original_mode != 'RGB':
            logger.info(f"Converting image from {original_mode} mode to RGB mode")
            image = image.convert('RGB')
        
        # Get image dimensions
        img_width, img_height = image.size
        img_pixels = img_width * img_height
        
        # Adjust image size
        min_size = params['min_size']
        max_size = params['max_size']
        
        # If image is smaller than minimum size, enlarge
        if img_pixels < (min_size * min_size):
            scale_factor = math.sqrt((min_size * min_size) / img_pixels)
            new_width = max(min_size, int(img_width * scale_factor))
            new_height = max(min_size, int(img_height * scale_factor))
            image = image.resize((new_width, new_height), Image.LANCZOS)
            logger.info(f"Image size smaller than minimum requirement, enlarged to {new_width}x{new_height}")
        
        # If image exceeds maximum size, reduce
        max_pixels = max_size * max_size
        if img_pixels > max_pixels:
            scale_factor = math.sqrt(max_pixels / img_pixels)
            new_width = min(max_size, int(img_width * scale_factor))
            new_height = min(max_size, int(img_height * scale_factor))
            image = image.resize((new_width, new_height), Image.LANCZOS)
            logger.info(f"Image size exceeds maximum limit, reduced to {new_width}x{new_height}")
        
        # If long edge exceeds max_size, scale
        if max(img_width, img_height) > max_size:
            if img_width >= img_height:
                new_width = max_size
                new_height = int(img_height * max_size / img_width)
            else:
                new_height = max_size
                new_width = int(img_width * max_size / img_height)
            image = image.resize((new_width, new_height), Image.LANCZOS)
            logger.info(f"Image long edge exceeds maximum limit, adjusted to {new_width}x{new_height}")
        
        return image
    
    def _convert_to_output_format(
        self, 
        image: Image.Image, 
        output_type: str
    ) -> Union[Image.Image, np.ndarray, 'torch.Tensor']:
        """
        Convert PIL image to requested output format
        
        Parameters:
            image: PIL image object
            output_type: Output type, 'pil', 'numpy' or 'torch'
            
        Returns:
            Converted image data
        """
        if output_type.lower() == 'pil':
            return image
        
        elif output_type.lower() == 'numpy':
            return np.array(image)
        
        elif output_type.lower() == 'torch':
            if not HAS_TORCH:
                raise ImportError("PyTorch must be installed to support torch output format")
            
            # Convert to numpy array, then convert to torch tensor
            np_array = np.array(image)
            
            # Adjust channel order: HWC (numpy) -> CHW (torch)
            if len(np_array.shape) == 3:
                np_array = np_array.transpose(2, 0, 1)
            
            # Normalize to [0,1]
            if np_array.dtype == np.uint8:
                np_array = np_array.astype(np.float32) / 255.0
            
            return torch.from_numpy(np_array)
        
        else:
            raise ValueError(f"Unsupported output format: {output_type}, supported formats are: 'pil', 'numpy', 'torch'")
    
    def save_image(
        self, 
        image: Union[Image.Image, np.ndarray, 'torch.Tensor'],
        output_path: str,
        format: Optional[str] = None,
        quality: int = 95
    ) -> str:
        """
        Save image to file
        
        Parameters:
            image: Image data
            output_path: Output path
            format: Output format, such as 'JPEG', 'PNG', etc., default inferred from file extension
            quality: JPEG compression quality
            
        Returns:
            Saved file path
        """
        # Ensure output directory exists
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Convert to PIL image
        if isinstance(image, np.ndarray):
            # If 3D array with channels first, convert to HWC format
            if len(image.shape) == 3 and image.shape[0] in [1, 3, 4]:
                image = image.transpose(1, 2, 0)
            
            # If float data, convert to uint8
            if image.dtype == np.float32 or image.dtype == np.float64:
                image = (image * 255).astype(np.uint8)
            
            pil_image = Image.fromarray(image)
        
        elif HAS_TORCH and isinstance(image, torch.Tensor):
            # Convert to numpy array
            np_array = image.detach().cpu().numpy()
            
            # If 3D tensor with channels first, convert to HWC format
            if len(np_array.shape) == 3 and np_array.shape[0] in [1, 3, 4]:
                np_array = np_array.transpose(1, 2, 0)
            
            # If float data, convert to uint8
            if np_array.dtype == np.float32 or np_array.dtype == np.float64:
                np_array = (np_array * 255).astype(np.uint8)
            
            pil_image = Image.fromarray(np_array)
        
        elif isinstance(image, Image.Image):
            pil_image = image
        
        else:
            raise TypeError(f"Unsupported image type: {type(image)}")
        
        # Determine save format
        if format is None:
            ext = os.path.splitext(output_path)[1].lower()
            if ext in ['.jpg', '.jpeg']:
                format = 'JPEG'
            elif ext == '.png':
                format = 'PNG'
            elif ext == '.gif':
                format = 'GIF'
            elif ext == '.webp':
                format = 'WEBP'
            elif ext == '.bmp':
                format = 'BMP'
            else:
                format = 'JPEG'  # Default format
        
        # Save image
        try:
            if format.upper() == 'JPEG':
                pil_image.save(output_path, format=format, quality=quality)
            else:
                pil_image.save(output_path, format=format)
            
            logger.info(f"Image saved to: {output_path}")
            return output_path
        
        except Exception as e:
            logger.error(f"Failed to save image: {str(e)}")
            raise
    
    def to_base64(
        self, 
        image: Union[Image.Image, np.ndarray, 'torch.Tensor'],
        format: str = 'JPEG',
        quality: int = 95,
        include_data_uri: bool = True
    ) -> str:
        """
        Convert image to Base64 string
        
        Parameters:
            image: Image data
            format: Output format, such as 'JPEG', 'PNG', etc.
            quality: JPEG compression quality
            include_data_uri: Whether to include data URI prefix
            
        Returns:
            Base64 encoded image string
        """
        # Convert to PIL image
        if isinstance(image, np.ndarray):
            # If 3D array with channels first, convert to HWC format
            if len(image.shape) == 3 and image.shape[0] in [1, 3, 4]:
                image = image.transpose(1, 2, 0)
            
            # If float data, convert to uint8
            if image.dtype == np.float32 or image.dtype == np.float64:
                image = (image * 255).astype(np.uint8)
            
            pil_image = Image.fromarray(image)
        
        elif HAS_TORCH and isinstance(image, torch.Tensor):
            # Convert to numpy array
            np_array = image.detach().cpu().numpy()
            
            # If 3D tensor with channels first, convert to HWC format
            if len(np_array.shape) == 3 and np_array.shape[0] in [1, 3, 4]:
                np_array = np_array.transpose(1, 2, 0)
            
            # If float data, convert to uint8
            if np_array.dtype == np.float32 or np_array.dtype == np.float64:
                np_array = (np_array * 255).astype(np.uint8)
            
            pil_image = Image.fromarray(np_array)
        
        elif isinstance(image, Image.Image):
            pil_image = image
        
        else:
            raise TypeError(f"Unsupported image type: {type(image)}")
        
        # Save image to memory buffer
        buffer = io.BytesIO()
        
        # Save according to format
        format = format.upper()
        if format == 'JPEG':
            # Ensure RGB mode
            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')
            pil_image.save(buffer, format=format, quality=quality)
        else:
            pil_image.save(buffer, format=format)
        
        # Convert to Base64
        encoded_string = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        # Add data URI prefix
        if include_data_uri:
            mime_type = format.lower()
            if mime_type == 'jpeg':
                mime_type = 'jpeg'
            return f"data:image/{mime_type};base64,{encoded_string}"
        
        return encoded_string


# Usage examples
def main():
    """Example usage"""
    converter = ImageConverter()
    
    # Example 1: Load from local file and convert to numpy array
    try:
        image = converter.process_image('example.jpg', output_type='numpy')
        print(f"Image shape: {image.shape}")
    except Exception as e:
        print(f"Processing failed: {e}")
    
    # Example 2: Load from URL and convert to PyTorch tensor
    try:
        image = converter.process_image(
            'https://example.com/image.jpg', 
            output_type='torch',
            min_size=224,
            max_size=1024
        )
        print(f"Tensor shape: {image.shape}")
    except Exception as e:
        print(f"Processing failed: {e}")
    
    # Example 3: Save as different format
    try:
        pil_image = converter.process_image('example.jpg')
        converter.save_image(pil_image, 'output.png', format='PNG')
    except Exception as e:
        print(f"Processing failed: {e}")
    
    # Example 4: Convert to Base64
    try:
        pil_image = converter.process_image('example.jpg')
        base64_str = converter.to_base64(pil_image, format='JPEG', quality=90)
        print(f"Base64 length: {len(base64_str)}")
    except Exception as e:
        print(f"Processing failed: {e}")


if __name__ == "__main__":
    main()