"""
Image Quality Validation System

Comprehensive image quality assessment including resolution, sharpness,
brightness, contrast, color balance, and noise detection.
"""

import asyncio
import logging
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
from dataclasses import dataclass
from PIL import Image, ImageStat, ImageFilter
import cv2
import io
import base64
import httpx

logger = logging.getLogger(__name__)


@dataclass
class QualityMetrics:
    """Image quality metrics data structure"""
    resolution_score: float = 0.0
    sharpness_score: float = 0.0
    brightness_score: float = 0.0
    contrast_score: float = 0.0
    color_balance_score: float = 0.0
    noise_score: float = 0.0
    overall_score: float = 0.0
    
    # Detailed metrics
    width: int = 0
    height: int = 0
    total_pixels: int = 0
    aspect_ratio: float = 0.0
    file_size: int = 0
    format: str = ""
    
    # Quality issues
    issues: List[str] = None
    recommendations: List[str] = None
    
    def __post_init__(self):
        if self.issues is None:
            self.issues = []
        if self.recommendations is None:
            self.recommendations = []


class ImageQualityValidator:
    """Image quality validation and assessment system"""
    
    def __init__(self):
        self.min_resolution = (512, 512)
        self.preferred_resolution = (1024, 1024)
        self.max_file_size = 10 * 1024 * 1024  # 10MB
        self.quality_thresholds = {
            'resolution': 70,
            'sharpness': 60,
            'brightness': 50,
            'contrast': 50,
            'color_balance': 60,
            'noise': 70,
            'overall': 65
        }
    
    async def validate_image(
        self,
        image_source: Union[str, Path, bytes],
        detailed: bool = True
    ) -> QualityMetrics:
        """Validate image quality from various sources"""
        
        # Load image based on source type
        if isinstance(image_source, str):
            if image_source.startswith(('http://', 'https://')):
                # URL
                image_array, pil_image, file_info = await self._load_from_url(image_source)
            elif image_source.startswith('data:'):
                # Base64 data URL
                image_array, pil_image, file_info = await self._load_from_base64(image_source)
            else:
                # File path
                image_array, pil_image, file_info = await self._load_from_path(Path(image_source))
        elif isinstance(image_source, Path):
            # Path object
            image_array, pil_image, file_info = await self._load_from_path(image_source)
        elif isinstance(image_source, bytes):
            # Raw bytes
            image_array, pil_image, file_info = await self._load_from_bytes(image_source)
        else:
            raise ValueError(f"Unsupported image source type: {type(image_source)}")
        
        # Initialize metrics
        metrics = QualityMetrics(
            width=file_info['width'],
            height=file_info['height'],
            total_pixels=file_info['width'] * file_info['height'],
            aspect_ratio=file_info['width'] / file_info['height'],
            file_size=file_info.get('file_size', 0),
            format=file_info.get('format', 'Unknown')
        )
        
        # Perform quality assessments
        if detailed:
            metrics.resolution_score = await self._assess_resolution(pil_image)
            metrics.sharpness_score = await self._assess_sharpness(image_array)
            metrics.brightness_score = await self._assess_brightness(pil_image)
            metrics.contrast_score = await self._assess_contrast(pil_image)
            metrics.color_balance_score = await self._assess_color_balance(pil_image)
            metrics.noise_score = await self._assess_noise(image_array)
        else:
            # Quick assessment
            metrics.resolution_score = await self._assess_resolution(pil_image)
            metrics.sharpness_score = await self._assess_sharpness_quick(pil_image)
            metrics.brightness_score = await self._assess_brightness(pil_image)
            metrics.contrast_score = await self._assess_contrast(pil_image)
        
        # Calculate overall score
        metrics.overall_score = await self._calculate_overall_score(metrics)
        
        # Identify issues and recommendations
        await self._identify_issues(metrics)
        await self._generate_recommendations(metrics)
        
        return metrics
    
    async def _load_from_url(self, url: str) -> tuple:
        """Load image from URL"""
        async with httpx.AsyncClient() as client:
            response = await client.get(url, timeout=30.0)
            response.raise_for_status()
            return await self._load_from_bytes(response.content)
    
    async def _load_from_base64(self, data_url: str) -> tuple:
        """Load image from base64 data URL"""
        if data_url.startswith('data:'):
            header, data = data_url.split(',', 1)
        else:
            data = data_url
        
        image_bytes = base64.b64decode(data)
        return await self._load_from_bytes(image_bytes)
    
    async def _load_from_path(self, file_path: Path) -> tuple:
        """Load image from file path"""
        with open(file_path, 'rb') as f:
            image_bytes = f.read()
        
        return await self._load_from_bytes(image_bytes, file_path.stat().st_size)
    
    async def _load_from_bytes(self, image_bytes: bytes, file_size: Optional[int] = None) -> tuple:
        """Load image from bytes"""
        # Load with PIL
        pil_image = Image.open(io.BytesIO(image_bytes))
        
        # Convert to numpy array for OpenCV
        if pil_image.mode == 'RGBA':
            # Convert RGBA to RGB
            rgb_image = Image.new('RGB', pil_image.size, (255, 255, 255))
            rgb_image.paste(pil_image, mask=pil_image.split()[-1])
            pil_image = rgb_image
        elif pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
        
        # Convert to numpy array (BGR for OpenCV)
        image_array = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        
        file_info = {
            'width': pil_image.width,
            'height': pil_image.height,
            'format': pil_image.format or 'Unknown',
            'file_size': file_size or len(image_bytes)
        }
        
        return image_array, pil_image, file_info
    
    async def _assess_resolution(self, pil_image: Image.Image) -> float:
        """Assess image resolution quality"""
        width, height = pil_image.size
        total_pixels = width * height
        
        # Score based on resolution
        if total_pixels >= self.preferred_resolution[0] * self.preferred_resolution[1]:
            score = 100
        elif total_pixels >= self.min_resolution[0] * self.min_resolution[1]:
            # Linear interpolation between min and preferred
            min_pixels = self.min_resolution[0] * self.min_resolution[1]
            pref_pixels = self.preferred_resolution[0] * self.preferred_resolution[1]
            score = 70 + 30 * (total_pixels - min_pixels) / (pref_pixels - min_pixels)
        else:
            # Below minimum resolution
            min_pixels = self.min_resolution[0] * self.min_resolution[1]
            score = max(0, 70 * total_pixels / min_pixels)
        
        return min(100, score)
    
    async def _assess_sharpness(self, image_array: np.ndarray) -> float:
        """Assess image sharpness using Laplacian variance"""
        # Convert to grayscale
        gray = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
        
        # Calculate Laplacian variance
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # Normalize to 0-100 scale (empirically determined thresholds)
        if laplacian_var >= 1000:
            score = 100
        elif laplacian_var >= 100:
            score = 60 + 40 * (laplacian_var - 100) / 900
        else:
            score = max(0, 60 * laplacian_var / 100)
        
        return min(100, score)
    
    async def _assess_sharpness_quick(self, pil_image: Image.Image) -> float:
        """Quick sharpness assessment using PIL"""
        # Apply edge detection filter
        edges = pil_image.filter(ImageFilter.FIND_EDGES)
        
        # Calculate edge strength
        stat = ImageStat.Stat(edges)
        edge_strength = sum(stat.mean) / len(stat.mean)
        
        # Normalize to 0-100 scale
        score = min(100, edge_strength * 2)
        
        return score
    
    async def _assess_brightness(self, pil_image: Image.Image) -> float:
        """Assess image brightness"""
        # Convert to grayscale for brightness calculation
        if pil_image.mode != 'L':
            gray = pil_image.convert('L')
        else:
            gray = pil_image
        
        # Calculate mean brightness
        stat = ImageStat.Stat(gray)
        brightness = stat.mean[0]  # 0-255 scale
        
        # Optimal brightness is around 128 (middle gray)
        # Score decreases as brightness moves away from optimal
        optimal = 128
        deviation = abs(brightness - optimal)
        
        if deviation <= 20:
            score = 100
        elif deviation <= 60:
            score = 80 - 30 * (deviation - 20) / 40
        else:
            score = max(0, 50 - 50 * (deviation - 60) / 67)
        
        return score
    
    async def _assess_contrast(self, pil_image: Image.Image) -> float:
        """Assess image contrast"""
        # Convert to grayscale
        if pil_image.mode != 'L':
            gray = pil_image.convert('L')
        else:
            gray = pil_image
        
        # Calculate standard deviation as contrast measure
        stat = ImageStat.Stat(gray)
        contrast = stat.stddev[0]
        
        # Normalize to 0-100 scale
        # Good contrast typically has stddev > 40
        if contrast >= 60:
            score = 100
        elif contrast >= 30:
            score = 70 + 30 * (contrast - 30) / 30
        else:
            score = max(0, 70 * contrast / 30)
        
        return min(100, score)
    
    async def _assess_color_balance(self, pil_image: Image.Image) -> float:
        """Assess color balance"""
        if pil_image.mode not in ('RGB', 'RGBA'):
            return 100  # Skip for non-color images
        
        # Calculate channel statistics
        stat = ImageStat.Stat(pil_image)
        
        if len(stat.mean) >= 3:
            r_mean, g_mean, b_mean = stat.mean[:3]
            
            # Calculate color balance
            total_mean = (r_mean + g_mean + b_mean) / 3
            
            # Calculate deviation from balanced
            r_dev = abs(r_mean - total_mean) / total_mean
            g_dev = abs(g_mean - total_mean) / total_mean
            b_dev = abs(b_mean - total_mean) / total_mean
            
            avg_deviation = (r_dev + g_dev + b_dev) / 3
            
            # Score based on deviation (lower deviation = better balance)
            if avg_deviation <= 0.1:
                score = 100
            elif avg_deviation <= 0.3:
                score = 80 - 30 * (avg_deviation - 0.1) / 0.2
            else:
                score = max(0, 50 - 50 * (avg_deviation - 0.3) / 0.7)
        else:
            score = 100
        
        return score
    
    async def _assess_noise(self, image_array: np.ndarray) -> float:
        """Assess image noise level"""
        # Convert to grayscale
        gray = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur and calculate difference
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        noise = cv2.absdiff(gray, blurred)
        
        # Calculate noise level
        noise_level = np.mean(noise)
        
        # Score based on noise level (lower noise = higher score)
        if noise_level <= 5:
            score = 100
        elif noise_level <= 15:
            score = 80 - 30 * (noise_level - 5) / 10
        else:
            score = max(0, 50 - 50 * (noise_level - 15) / 35)
        
        return score
    
    async def _calculate_overall_score(self, metrics: QualityMetrics) -> float:
        """Calculate weighted overall quality score"""
        weights = {
            'resolution': 0.25,
            'sharpness': 0.25,
            'brightness': 0.15,
            'contrast': 0.15,
            'color_balance': 0.10,
            'noise': 0.10
        }
        
        overall = (
            metrics.resolution_score * weights['resolution'] +
            metrics.sharpness_score * weights['sharpness'] +
            metrics.brightness_score * weights['brightness'] +
            metrics.contrast_score * weights['contrast'] +
            metrics.color_balance_score * weights['color_balance'] +
            metrics.noise_score * weights['noise']
        )
        
        return round(overall, 1)
    
    async def _identify_issues(self, metrics: QualityMetrics):
        """Identify quality issues"""
        issues = []
        
        if metrics.resolution_score < self.quality_thresholds['resolution']:
            issues.append("Low resolution")
        
        if metrics.sharpness_score < self.quality_thresholds['sharpness']:
            issues.append("Image appears blurry or out of focus")
        
        if metrics.brightness_score < self.quality_thresholds['brightness']:
            issues.append("Poor brightness levels (too dark or too bright)")
        
        if metrics.contrast_score < self.quality_thresholds['contrast']:
            issues.append("Low contrast")
        
        if metrics.color_balance_score < self.quality_thresholds['color_balance']:
            issues.append("Color imbalance detected")
        
        if metrics.noise_score < self.quality_thresholds['noise']:
            issues.append("High noise levels")
        
        if metrics.file_size > self.max_file_size:
            issues.append("File size too large")
        
        # Aspect ratio issues
        if metrics.aspect_ratio < 0.5 or metrics.aspect_ratio > 2.0:
            issues.append("Unusual aspect ratio")
        
        metrics.issues = issues
    
    async def _generate_recommendations(self, metrics: QualityMetrics):
        """Generate improvement recommendations"""
        recommendations = []
        
        if metrics.resolution_score < self.quality_thresholds['resolution']:
            recommendations.append("Use higher resolution source images")
        
        if metrics.sharpness_score < self.quality_thresholds['sharpness']:
            recommendations.append("Apply sharpening filter or use better focus")
        
        if metrics.brightness_score < 30:
            recommendations.append("Increase image brightness")
        elif metrics.brightness_score < 50 and metrics.brightness_score >= 30:
            recommendations.append("Adjust brightness levels")
        
        if metrics.contrast_score < self.quality_thresholds['contrast']:
            recommendations.append("Increase contrast to improve image definition")
        
        if metrics.color_balance_score < self.quality_thresholds['color_balance']:
            recommendations.append("Adjust color balance for more natural colors")
        
        if metrics.noise_score < self.quality_thresholds['noise']:
            recommendations.append("Apply noise reduction filter")
        
        if metrics.file_size > self.max_file_size:
            recommendations.append("Compress image to reduce file size")
        
        if not recommendations:
            recommendations.append("Image quality is good")
        
        metrics.recommendations = recommendations
    
    def get_quality_grade(self, overall_score: float) -> str:
        """Get quality grade based on overall score"""
        if overall_score >= 90:
            return "Excellent"
        elif overall_score >= 80:
            return "Good"
        elif overall_score >= 70:
            return "Fair"
        elif overall_score >= 60:
            return "Poor"
        else:
            return "Very Poor"
    
    async def batch_validate(
        self,
        image_sources: List[Union[str, Path, bytes]],
        detailed: bool = False
    ) -> List[QualityMetrics]:
        """Validate multiple images in batch"""
        tasks = [
            self.validate_image(source, detailed)
            for source in image_sources
        ]
        
        return await asyncio.gather(*tasks, return_exceptions=True)