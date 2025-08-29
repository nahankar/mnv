"""
Enhanced Image Processing Pipeline

Advanced image processing capabilities including resizing, optimization,
watermarking, thumbnail generation, and format conversion for different platforms.
"""

import asyncio
import logging
import os
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union
import tempfile
import aiofiles
from PIL import Image, ImageDraw, ImageFont, ImageEnhance, ImageFilter, ImageOps
import io
import base64

logger = logging.getLogger(__name__)


class ImageProcessor:
    """Enhanced image processing pipeline"""
    
    def __init__(self):
        self.supported_formats = ['JPEG', 'PNG', 'WEBP']
        self.platform_specs = {
            'youtube': {
                'thumbnail': (1280, 720),
                'video_frame': (1920, 1080),
                'aspect_ratios': ['16:9']
            },
            'instagram': {
                'square': (1080, 1080),
                'story': (1080, 1920),
                'reel': (1080, 1920),
                'aspect_ratios': ['1:1', '9:16']
            },
            'tiktok': {
                'video': (1080, 1920),
                'aspect_ratios': ['9:16']
            },
            'facebook': {
                'post': (1200, 630),
                'story': (1080, 1920),
                'aspect_ratios': ['16:9', '1:1', '9:16']
            }
        }
    
    async def download_and_save_image(self, image_url: str, file_path: Path) -> Dict[str, Any]:
        """Download image from URL and save to file"""
        import httpx
        
        async with httpx.AsyncClient() as client:
            response = await client.get(image_url, timeout=30.0)
            response.raise_for_status()
            
            async with aiofiles.open(file_path, 'wb') as f:
                await f.write(response.content)
            
            # Get image metadata
            with Image.open(file_path) as img:
                return {
                    "format": img.format,
                    "size": img.size,
                    "mode": img.mode,
                    "file_size": file_path.stat().st_size,
                    "has_transparency": img.mode in ('RGBA', 'LA') or 'transparency' in img.info
                }
    
    async def save_base64_image(self, image_data: str, file_path: Path) -> Dict[str, Any]:
        """Save base64 image data to file"""
        # Handle data URL format
        if image_data.startswith('data:'):
            header, image_data = image_data.split(',', 1)
        
        # Decode base64 data
        image_bytes = base64.b64decode(image_data)
        
        async with aiofiles.open(file_path, 'wb') as f:
            await f.write(image_bytes)
        
        # Get image metadata
        with Image.open(file_path) as img:
            return {
                "format": img.format,
                "size": img.size,
                "mode": img.mode,
                "file_size": file_path.stat().st_size,
                "has_transparency": img.mode in ('RGBA', 'LA') or 'transparency' in img.info
            }
    
    async def optimize_image(
        self,
        file_path: Path,
        quality: int = 85,
        target_format: str = 'JPEG',
        max_file_size: Optional[int] = None
    ) -> Dict[str, Any]:
        """Optimize image for web delivery with advanced options"""
        
        with Image.open(file_path) as img:
            original_size = file_path.stat().st_size
            
            # Convert to RGB if targeting JPEG
            if target_format == 'JPEG' and img.mode in ('RGBA', 'LA', 'P'):
                # Create white background for transparency
                background = Image.new('RGB', img.size, (255, 255, 255))
                if img.mode == 'P':
                    img = img.convert('RGBA')
                background.paste(img, mask=img.split()[-1] if img.mode == 'RGBA' else None)
                img = background
            
            # Determine output path
            if target_format == 'JPEG':
                optimized_path = file_path.with_suffix('.jpg')
            elif target_format == 'PNG':
                optimized_path = file_path.with_suffix('.png')
            elif target_format == 'WEBP':
                optimized_path = file_path.with_suffix('.webp')
            else:
                optimized_path = file_path
            
            # Save with optimization
            save_kwargs = {'optimize': True}
            
            if target_format == 'JPEG':
                save_kwargs.update({
                    'quality': quality,
                    'progressive': True
                })
            elif target_format == 'PNG':
                save_kwargs.update({
                    'compress_level': 6
                })
            elif target_format == 'WEBP':
                save_kwargs.update({
                    'quality': quality,
                    'method': 6
                })
            
            # If max file size is specified, iteratively reduce quality
            if max_file_size:
                current_quality = quality
                while current_quality > 10:
                    # Save to temporary buffer to check size
                    buffer = io.BytesIO()
                    img.save(buffer, format=target_format, **{**save_kwargs, 'quality': current_quality})
                    
                    if len(buffer.getvalue()) <= max_file_size:
                        break
                    
                    current_quality -= 10
                
                save_kwargs['quality'] = current_quality
            
            img.save(optimized_path, format=target_format, **save_kwargs)
            
            # Remove original if different format
            if optimized_path != file_path and file_path.exists():
                file_path.unlink()
            
            optimized_size = optimized_path.stat().st_size
            
            return {
                "optimized_path": optimized_path,
                "file_size": optimized_size,
                "original_size": original_size,
                "compression_ratio": original_size / optimized_size if optimized_size > 0 else 1,
                "format": target_format,
                "quality_used": save_kwargs.get('quality', quality)
            }
    
    async def resize_image(
        self,
        file_path: Path,
        target_size: Tuple[int, int],
        maintain_aspect: bool = True,
        crop_mode: str = 'center',
        upscale: bool = False
    ) -> Path:
        """Resize image with advanced options"""
        
        with Image.open(file_path) as img:
            original_size = img.size
            
            # Don't upscale unless explicitly requested
            if not upscale:
                if target_size[0] > original_size[0] or target_size[1] > original_size[1]:
                    target_size = (
                        min(target_size[0], original_size[0]),
                        min(target_size[1], original_size[1])
                    )
            
            if maintain_aspect:
                # Calculate aspect-preserving size
                img.thumbnail(target_size, Image.Resampling.LANCZOS)
                resized_img = img
            else:
                if crop_mode == 'center':
                    # Crop to exact size from center
                    resized_img = ImageOps.fit(img, target_size, Image.Resampling.LANCZOS)
                else:
                    # Simple resize (may distort)
                    resized_img = img.resize(target_size, Image.Resampling.LANCZOS)
            
            # Save resized image
            resized_path = file_path.with_name(f"{file_path.stem}_resized{file_path.suffix}")
            resized_img.save(resized_path)
            
            return resized_path
    
    async def create_thumbnail(
        self,
        file_path: Path,
        size: Tuple[int, int] = (300, 300),
        quality: int = 80
    ) -> Path:
        """Create thumbnail with consistent quality"""
        
        with Image.open(file_path) as img:
            # Create thumbnail maintaining aspect ratio
            img.thumbnail(size, Image.Resampling.LANCZOS)
            
            # Create thumbnail path
            thumb_path = file_path.with_name(f"{file_path.stem}_thumb{file_path.suffix}")
            
            # Save thumbnail
            if img.mode in ('RGBA', 'LA'):
                img.save(thumb_path, 'PNG', optimize=True)
            else:
                img.save(thumb_path, 'JPEG', quality=quality, optimize=True)
            
            return thumb_path
    
    async def add_watermark(
        self,
        file_path: Path,
        watermark_text: str,
        position: str = 'bottom-right',
        opacity: float = 0.7,
        font_size: int = 36
    ) -> Path:
        """Add text watermark to image"""
        
        with Image.open(file_path) as img:
            # Convert to RGBA for transparency support
            if img.mode != 'RGBA':
                img = img.convert('RGBA')
            
            # Create transparent overlay
            overlay = Image.new('RGBA', img.size, (255, 255, 255, 0))
            draw = ImageDraw.Draw(overlay)
            
            # Try to load a font, fallback to default
            try:
                font = ImageFont.truetype("arial.ttf", font_size)
            except (OSError, IOError):
                try:
                    font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", font_size)
                except (OSError, IOError):
                    font = ImageFont.load_default()
            
            # Get text dimensions
            bbox = draw.textbbox((0, 0), watermark_text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            
            # Calculate position
            margin = 20
            if position == 'bottom-right':
                x = img.width - text_width - margin
                y = img.height - text_height - margin
            elif position == 'bottom-left':
                x = margin
                y = img.height - text_height - margin
            elif position == 'top-right':
                x = img.width - text_width - margin
                y = margin
            elif position == 'top-left':
                x = margin
                y = margin
            else:  # center
                x = (img.width - text_width) // 2
                y = (img.height - text_height) // 2
            
            # Draw text with opacity
            text_color = (255, 255, 255, int(255 * opacity))
            draw.text((x, y), watermark_text, font=font, fill=text_color)
            
            # Composite overlay onto image
            watermarked = Image.alpha_composite(img, overlay)
            
            # Convert back to original mode if needed
            if file_path.suffix.lower() in ['.jpg', '.jpeg']:
                watermarked = watermarked.convert('RGB')
            
            # Save watermarked image
            watermarked_path = file_path.with_name(f"{file_path.stem}_watermarked{file_path.suffix}")
            watermarked.save(watermarked_path)
            
            return watermarked_path
    
    async def create_platform_variants(
        self,
        file_path: Path,
        platforms: List[str]
    ) -> Dict[str, List[Path]]:
        """Create optimized variants for different platforms"""
        
        variants = {}
        
        for platform in platforms:
            if platform not in self.platform_specs:
                logger.warning(f"Unknown platform: {platform}")
                continue
            
            platform_variants = []
            specs = self.platform_specs[platform]
            
            # Create variants for each size in platform spec
            for variant_name, size in specs.items():
                if variant_name == 'aspect_ratios':
                    continue
                
                try:
                    # Resize for platform
                    variant_path = await self.resize_image(
                        file_path,
                        size,
                        maintain_aspect=False,
                        crop_mode='center'
                    )
                    
                    # Optimize for platform
                    optimized = await self.optimize_image(
                        variant_path,
                        quality=85,
                        target_format='JPEG'
                    )
                    
                    # Rename to include platform and variant
                    final_path = file_path.with_name(
                        f"{file_path.stem}_{platform}_{variant_name}{file_path.suffix}"
                    )
                    optimized['optimized_path'].rename(final_path)
                    
                    platform_variants.append(final_path)
                    
                except Exception as e:
                    logger.error(f"Failed to create {platform} {variant_name} variant: {e}")
            
            variants[platform] = platform_variants
        
        return variants
    
    async def enhance_image(
        self,
        file_path: Path,
        brightness: float = 1.0,
        contrast: float = 1.0,
        saturation: float = 1.0,
        sharpness: float = 1.0
    ) -> Path:
        """Enhance image with brightness, contrast, saturation, and sharpness"""
        
        with Image.open(file_path) as img:
            enhanced = img
            
            # Apply enhancements
            if brightness != 1.0:
                enhancer = ImageEnhance.Brightness(enhanced)
                enhanced = enhancer.enhance(brightness)
            
            if contrast != 1.0:
                enhancer = ImageEnhance.Contrast(enhanced)
                enhanced = enhancer.enhance(contrast)
            
            if saturation != 1.0:
                enhancer = ImageEnhance.Color(enhanced)
                enhanced = enhancer.enhance(saturation)
            
            if sharpness != 1.0:
                enhancer = ImageEnhance.Sharpness(enhanced)
                enhanced = enhancer.enhance(sharpness)
            
            # Save enhanced image
            enhanced_path = file_path.with_name(f"{file_path.stem}_enhanced{file_path.suffix}")
            enhanced.save(enhanced_path)
            
            return enhanced_path
    
    async def apply_filter(
        self,
        file_path: Path,
        filter_type: str = 'none'
    ) -> Path:
        """Apply artistic filters to image"""
        
        with Image.open(file_path) as img:
            filtered = img
            
            if filter_type == 'blur':
                filtered = img.filter(ImageFilter.GaussianBlur(radius=2))
            elif filter_type == 'sharpen':
                filtered = img.filter(ImageFilter.SHARPEN)
            elif filter_type == 'edge_enhance':
                filtered = img.filter(ImageFilter.EDGE_ENHANCE)
            elif filter_type == 'emboss':
                filtered = img.filter(ImageFilter.EMBOSS)
            elif filter_type == 'smooth':
                filtered = img.filter(ImageFilter.SMOOTH)
            elif filter_type == 'detail':
                filtered = img.filter(ImageFilter.DETAIL)
            
            # Save filtered image
            filtered_path = file_path.with_name(f"{file_path.stem}_filtered{file_path.suffix}")
            filtered.save(filtered_path)
            
            return filtered_path
    
    async def create_collage(
        self,
        image_paths: List[Path],
        layout: str = 'grid',
        output_size: Tuple[int, int] = (1920, 1080),
        spacing: int = 10
    ) -> Path:
        """Create a collage from multiple images"""
        
        if not image_paths:
            raise ValueError("No images provided for collage")
        
        # Load all images
        images = []
        for path in image_paths:
            with Image.open(path) as img:
                images.append(img.copy())
        
        # Create collage based on layout
        if layout == 'grid':
            # Calculate grid dimensions
            num_images = len(images)
            cols = int(num_images ** 0.5)
            rows = (num_images + cols - 1) // cols
            
            # Calculate cell size
            cell_width = (output_size[0] - spacing * (cols + 1)) // cols
            cell_height = (output_size[1] - spacing * (rows + 1)) // rows
            
            # Create collage canvas
            collage = Image.new('RGB', output_size, (255, 255, 255))
            
            # Place images
            for i, img in enumerate(images):
                row = i // cols
                col = i % cols
                
                # Resize image to fit cell
                img_resized = ImageOps.fit(img, (cell_width, cell_height), Image.Resampling.LANCZOS)
                
                # Calculate position
                x = spacing + col * (cell_width + spacing)
                y = spacing + row * (cell_height + spacing)
                
                collage.paste(img_resized, (x, y))
        
        elif layout == 'horizontal':
            # Arrange images horizontally
            total_width = sum(img.width for img in images) + spacing * (len(images) + 1)
            max_height = max(img.height for img in images) + spacing * 2
            
            collage = Image.new('RGB', (total_width, max_height), (255, 255, 255))
            
            x_offset = spacing
            for img in images:
                y_offset = (max_height - img.height) // 2
                collage.paste(img, (x_offset, y_offset))
                x_offset += img.width + spacing
        
        elif layout == 'vertical':
            # Arrange images vertically
            max_width = max(img.width for img in images) + spacing * 2
            total_height = sum(img.height for img in images) + spacing * (len(images) + 1)
            
            collage = Image.new('RGB', (max_width, total_height), (255, 255, 255))
            
            y_offset = spacing
            for img in images:
                x_offset = (max_width - img.width) // 2
                collage.paste(img, (x_offset, y_offset))
                y_offset += img.height + spacing
        
        # Save collage
        collage_path = image_paths[0].with_name(f"collage_{layout}_{len(images)}_images.jpg")
        collage.save(collage_path, 'JPEG', quality=90)
        
        return collage_path
    
    async def get_image_info(self, file_path: Path) -> Dict[str, Any]:
        """Get comprehensive image information"""
        
        with Image.open(file_path) as img:
            # Basic info
            info = {
                'filename': file_path.name,
                'format': img.format,
                'mode': img.mode,
                'size': img.size,
                'width': img.width,
                'height': img.height,
                'aspect_ratio': round(img.width / img.height, 2),
                'file_size': file_path.stat().st_size,
                'has_transparency': img.mode in ('RGBA', 'LA') or 'transparency' in img.info
            }
            
            # EXIF data if available
            if hasattr(img, '_getexif') and img._getexif():
                info['exif'] = dict(img._getexif())
            
            # Color analysis
            if img.mode in ('RGB', 'RGBA'):
                # Get dominant colors
                img_small = img.resize((50, 50))
                colors = img_small.getcolors(maxcolors=256*256*256)
                if colors:
                    dominant_color = max(colors, key=lambda x: x[0])[1]
                    info['dominant_color'] = dominant_color
            
            return info