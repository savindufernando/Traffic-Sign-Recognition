"""
Research-Grade Synthetic Data Generator for Sri Lankan Traffic Signs.

Features:
- Context-aware sign placement
- Depth/scale consistency using vanishing point
- Automatic realism filtering
- Hard negative sample generation
- Sri Lanka-specific environmental effects
- Full metadata logging for reproducibility
"""

import cv2
import numpy as np
import json
import random
import logging
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional, Tuple, List, Dict, Any
from enum import Enum
from datetime import datetime

try:
    import albumentations as A
    ALBUMENTATIONS_AVAILABLE = True
except ImportError:
    ALBUMENTATIONS_AVAILABLE = False
    print("Warning: albumentations not available, some effects will be limited")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Custom JSON encoder for numpy types
class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles numpy types."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


# =============================================================================
# ENUMS AND DATA CLASSES
# =============================================================================

class RoadContext(Enum):
    """Road context types for context-aware placement."""
    HIGHWAY = "highway"
    URBAN = "urban"
    RESIDENTIAL = "residential"
    SCHOOL_ZONE = "school_zone"
    RURAL = "rural"
    COASTAL = "coastal"


class WeatherCondition(Enum):
    """Weather conditions for augmentation."""
    CLEAR = "clear"
    RAIN = "rain"
    FOG = "fog"
    DUST = "dust"
    HARSH_SUN = "harsh_sun"


class TimeOfDay(Enum):
    """Time of day for lighting conditions."""
    MORNING = "morning"
    NOON = "noon"
    GOLDEN_HOUR = "golden_hour"
    NIGHT = "night"


@dataclass
class SignCategory:
    """Mapping of sign categories to appropriate road contexts."""
    name: str
    folder: str
    suitable_contexts: List[RoadContext]
    
    
# Define sign category to road context mappings
SIGN_CONTEXT_MAPPING = {
    "warning_signs": SignCategory(
        name="Warning Signs",
        folder="warning_signs",
        suitable_contexts=[RoadContext.HIGHWAY, RoadContext.URBAN, RoadContext.RURAL]
    ),
    "prohibitory_signs": SignCategory(
        name="Prohibitory Signs", 
        folder="regulatory_signs/prohibitory_signs",
        suitable_contexts=[RoadContext.HIGHWAY, RoadContext.URBAN, RoadContext.RESIDENTIAL]
    ),
    "mandatory_signs": SignCategory(
        name="Mandatory Signs",
        folder="regulatory_signs/mandatory_signs",
        suitable_contexts=[RoadContext.HIGHWAY, RoadContext.URBAN]
    ),
    "restrictive_signs": SignCategory(
        name="Restrictive Signs",
        folder="regulatory_signs/restrictive_signs",
        suitable_contexts=[RoadContext.HIGHWAY, RoadContext.URBAN, RoadContext.SCHOOL_ZONE]
    ),
    "priority_signs": SignCategory(
        name="Priority Signs",
        folder="regulatory_signs/priority_signs",
        suitable_contexts=[RoadContext.HIGHWAY, RoadContext.URBAN, RoadContext.RURAL]
    ),
    "traffic_light_signals": SignCategory(
        name="Traffic Lights",
        folder="traffic_light_signals",
        suitable_contexts=[RoadContext.URBAN, RoadContext.HIGHWAY]
    ),
    "other_signs": SignCategory(
        name="Information Signs",
        folder="directional_informative_signs/other_signs",
        suitable_contexts=[RoadContext.URBAN, RoadContext.HIGHWAY, RoadContext.RESIDENTIAL]
    ),
}


@dataclass
class ImageMetadata:
    """Metadata for each generated image."""
    source_template: str
    background_file: str
    weather_condition: str
    time_of_day: str
    road_context: str
    sign_position: Tuple[int, int]
    sign_scale: float
    perspective_applied: bool
    augmentations: List[str]
    bbox: Tuple[int, int, int, int]  # x, y, w, h
    random_seed: int
    generation_timestamp: str
    is_hard_negative: bool = False


# =============================================================================
# DEPTH AND SCALE MANAGER
# =============================================================================

class DepthScaleManager:
    """
    Handles depth-aware scaling of signs based on position in image.
    Uses simple vanishing point heuristics for perspective-correct sizing.
    """
    
    def __init__(self, image_height: int, image_width: int):
        self.image_height = image_height
        self.image_width = image_width
        # Vanishing point typically at horizon (upper third of image)
        self.vanishing_point_y = image_height * 0.35
        
    def get_scale_for_position(
        self, 
        y_position: int,
        base_scale: float = 0.15
    ) -> float:
        """
        Calculate appropriate scale based on vertical position.
        Objects closer to vanishing point appear smaller.
        
        Args:
            y_position: Y coordinate where sign will be placed
            base_scale: Base scale factor (proportion of image height)
            
        Returns:
            Adjusted scale factor
        """
        # Normalize position relative to vanishing point
        # At vanishing point: scale = 0.3x base
        # At bottom of image: scale = 1.5x base
        distance_from_vp = abs(y_position - self.vanishing_point_y)
        max_distance = self.image_height - self.vanishing_point_y
        
        # Linear interpolation with bounds
        scale_factor = 0.3 + (distance_from_vp / max_distance) * 1.2
        scale_factor = np.clip(scale_factor, 0.3, 1.5)
        
        return base_scale * scale_factor
    
    def get_valid_placement_zone(self) -> Tuple[int, int, int, int]:
        """
        Get the valid zone for sign placement (roadside areas).
        
        Returns:
            (x_min, y_min, x_max, y_max) bounds
        """
        # Signs typically appear in left or right third, not in center
        # Y position: from horizon to 80% of image height
        x_zones = [
            (0, int(self.image_width * 0.45)),  # Widen left zone
            (int(self.image_width * 0.55), self.image_width)  # Widen right zone
        ]
        y_min = int(self.vanishing_point_y)
        # Reduced from 0.85 to 0.65 to avoid placing signs on the dashboard/hood
        y_max = int(self.image_height * 0.65)
        
        return x_zones, y_min, y_max

    def get_y_range_for_scale(
        self, 
        min_scale_factor: float, 
        max_scale_factor: float
    ) -> Tuple[int, int]:
        """
        Calculate valid Y range for a given range of scale factors.
        Inverse of get_scale_for_position.
        """
        # Constraints of the current scaling model
        MODEL_MIN = 0.3
        MODEL_MAX = 1.5
        
        # Clip requested range to model capabilities
        target_min = max(MODEL_MIN, min_scale_factor)
        target_max = min(MODEL_MAX, max_scale_factor)
        
        if target_min > target_max:
            return self.vanishing_point_y, self.vanishing_point_y
            
        # Formula: scale_factor = 0.3 + (dist / max_dist) * 1.2
        # Inverse: dist = ((scale_factor - 0.3) / 1.2) * max_dist
        
        max_dist = self.image_height - self.vanishing_point_y
        
        dist_min = ((target_min - 0.3) / 1.2) * max_dist
        dist_max = ((target_max - 0.3) / 1.2) * max_dist
        
        y_start = int(self.vanishing_point_y + dist_min)
        y_end = int(self.vanishing_point_y + dist_max)
        
        return y_start, y_end


# =============================================================================
# CONTEXT-AWARE COMPOSITOR
# =============================================================================

class ContextAwareCompositor:
    """
    Handles context-aware sign placement and compositing.
    Places signs appropriately based on sign type and road context.
    """
    
    def __init__(self, seed: Optional[int] = None):
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
            
    def get_matching_context(
        self, 
        sign_category: str,
        background_folder: str
    ) -> bool:
        """
        Check if sign category matches background context.
        
        Args:
            sign_category: Category of the sign
            background_folder: Folder name of the background
            
        Returns:
            True if sign is appropriate for this context
        """
        # Map background folder to context
        folder_context_map = {
            "urban_day": RoadContext.URBAN,
            "rain": RoadContext.URBAN,
            "night": RoadContext.URBAN,
            "highway": RoadContext.HIGHWAY,
            "school_residential": RoadContext.SCHOOL_ZONE,
            "golden_hour": RoadContext.URBAN,
            "fog_mist": RoadContext.RURAL,
            "coastal": RoadContext.COASTAL,
            "rural": RoadContext.RURAL,
            "harsh_conditions": RoadContext.URBAN,
        }
        
        bg_context = folder_context_map.get(background_folder, RoadContext.URBAN)
        
        if sign_category in SIGN_CONTEXT_MAPPING:
            sign_info = SIGN_CONTEXT_MAPPING[sign_category]
            return bg_context in sign_info.suitable_contexts
        
        return True  # Default: allow placement
    
    def composite_sign(
        self,
        background: np.ndarray,
        sign: np.ndarray,
        position: Tuple[int, int],
        scale: float,
        apply_perspective: bool = True,
        light_direction: float = 0.0  # Angle in radians
    ) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
        """
        Composite sign onto background with effects.
        
        Args:
            background: Background image (H, W, 3)
            sign: Sign image with alpha channel (H, W, 4)
            position: (x, y) position for sign center
            scale: Scale factor for sign
            apply_perspective: Apply perspective distortion
            light_direction: Direction of light for shadow
            
        Returns:
            (composite_image, bounding_box)
        """
        bg_h, bg_w = background.shape[:2]
        
        # Resize sign
        sign_h, sign_w = sign.shape[:2]
        new_h = int(bg_h * scale)
        new_w = int(new_h * (sign_w / sign_h))
        sign_resized = cv2.resize(sign, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # Apply perspective if enabled
        if apply_perspective:
            sign_resized = self._apply_perspective(sign_resized)
            new_h, new_w = sign_resized.shape[:2]
        
        # Calculate position (ensure sign stays within bounds)
        x = int(position[0] - new_w // 2)
        y = int(position[1] - new_h // 2)
        
        # Clip to image bounds
        x = np.clip(x, 0, bg_w - new_w)
        y = np.clip(y, 0, bg_h - new_h)
        
        # Create output image
        output = background.copy()
        
        # Handle alpha channel
        if sign_resized.shape[2] == 4:
            alpha = sign_resized[:, :, 3:4] / 255.0
            sign_rgb = sign_resized[:, :, :3]
        else:
            alpha = np.ones((new_h, new_w, 1))
            sign_rgb = sign_resized
        
        # Add shadow
        output = self._add_shadow(output, alpha, x, y, light_direction)
        
        # Blend sign onto background
        roi = output[y:y+new_h, x:x+new_w]
        blended = (sign_rgb * alpha + roi * (1 - alpha)).astype(np.uint8)
        output[y:y+new_h, x:x+new_w] = blended
        
        # Return bounding box (x, y, w, h)
        bbox = (x, y, new_w, new_h)
        
        return output, bbox
    
    def _apply_perspective(self, image: np.ndarray) -> np.ndarray:
        """Apply subtle perspective distortion."""
        h, w = image.shape[:2]
        
        # Random perspective parameters
        # Random perspective parameters
        # Increased skew range for more variety (was 0.05)
        skew = np.random.uniform(-0.15, 0.15)
        
        # Define source points (corners of image)
        src_pts = np.float32([
            [0, 0], [w, 0], [w, h], [0, h]
        ])
        
        # Define destination points with perspective
        dst_pts = np.float32([
            [w * skew, h * abs(skew)],
            [w * (1 - skew), h * abs(skew)],
            [w * (1 + skew), h * (1 - abs(skew))],
            [-w * skew, h * (1 - abs(skew))]
        ])
        
        # Compute and apply perspective transform
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        warped = cv2.warpPerspective(image, M, (w, h), borderMode=cv2.BORDER_CONSTANT)
        
        return warped
    
    def _add_shadow(
        self, 
        image: np.ndarray, 
        alpha: np.ndarray,
        x: int, y: int,
        light_direction: float
    ) -> np.ndarray:
        """Add realistic shadow under the sign."""
        h, w = alpha.shape[:2]
        
        # Shadow offset based on light direction
        shadow_offset_x = int(w * 0.05 * np.cos(light_direction))
        shadow_offset_y = int(h * 0.05 * np.sin(light_direction)) + int(h * 0.02)
        
        # Create shadow mask
        shadow = (alpha[:, :, 0] * 0.3).astype(np.uint8)
        shadow = cv2.GaussianBlur(shadow, (15, 15), 0)
        
        # Apply shadow
        shadow_x = x + shadow_offset_x
        shadow_y = y + shadow_offset_y
        
        if 0 <= shadow_y < image.shape[0] - h and 0 <= shadow_x < image.shape[1] - w:
            roi = image[shadow_y:shadow_y+h, shadow_x:shadow_x+w]
            shadow_3d = np.stack([shadow] * 3, axis=-1) / 255.0
            image[shadow_y:shadow_y+h, shadow_x:shadow_x+w] = (
                roi * (1 - shadow_3d * 0.3)
            ).astype(np.uint8)
        
        return image


# =============================================================================
# REALISM FILTER
# =============================================================================

class RealismFilter:
    """
    Filters out unrealistic compositions.
    Rejects images where signs are poorly placed or blend poorly.
    """
    
    def __init__(
        self,
        min_sign_size: float = 0.01,  # Reduced: allow smaller signs (1% of image)
        max_sign_size: float = 0.5,   # Increased: allow larger signs
        min_contrast_ratio: float = 1.1  # Reduced: be less strict about contrast
    ):
        self.min_sign_size = min_sign_size
        self.max_sign_size = max_sign_size
        self.min_contrast_ratio = min_contrast_ratio
        
    def check_composition(
        self,
        composite: np.ndarray,
        bbox: Tuple[int, int, int, int],
        background: np.ndarray
    ) -> Tuple[bool, str]:
        """
        Check if composition is realistic.
        
        Returns:
            (is_valid, rejection_reason)
        """
        img_h, img_w = composite.shape[:2]
        x, y, w, h = bbox
        
        # Check 1: Sign size
        sign_area = w * h
        image_area = img_h * img_w
        size_ratio = sign_area / image_area
        
        if size_ratio < self.min_sign_size:
            return False, "sign_too_small"
        if size_ratio > self.max_sign_size:
            return False, "sign_too_large"
        
        # Check 2: Sign not in sky (upper 20% of image - relaxed for dashcam views)
        if y + h < img_h * 0.2:
            return False, "sign_in_sky"
        
        # Check 3: Contrast check
        sign_region = composite[y:y+h, x:x+w]
        sign_mean = np.mean(sign_region)
        
        # Sample background around the sign
        bg_sample = background[
            max(0, y-20):min(img_h, y+h+20),
            max(0, x-20):min(img_w, x+w+20)
        ]
        bg_mean = np.mean(bg_sample)
        
        contrast_ratio = max(sign_mean, bg_mean) / (min(sign_mean, bg_mean) + 1e-6)
        
        if contrast_ratio < self.min_contrast_ratio:
            return False, "low_contrast"
        
        return True, "valid"


# =============================================================================
# SRI LANKA ENVIRONMENTAL EFFECTS
# =============================================================================

class SriLankaEffects:
    """
    Applies Sri Lanka-specific environmental effects.
    Includes tropical sun, dust, monsoon rain, high humidity haze.
    """
    
    def __init__(self, seed: Optional[int] = None):
        if seed is not None:
            np.random.seed(seed)
            
    def apply_tropical_sun(self, image: np.ndarray) -> np.ndarray:
        """Apply harsh tropical sun glare effect."""
        # Increase brightness and add yellow tint
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] = hsv[:, :, 1] * 0.9  # Slightly desaturate
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * 1.2, 0, 255)  # Increase brightness
        
        image = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        
        # Add sun glare in random corner
        h, w = image.shape[:2]
        glare_x = np.random.choice([w * 0.1, w * 0.9])
        glare_y = h * 0.1
        
        Y, X = np.ogrid[:h, :w]
        dist = np.sqrt((X - glare_x)**2 + (Y - glare_y)**2)
        max_dist = np.sqrt(h**2 + w**2) * 0.3
        
        glare_mask = np.clip(1 - (dist / max_dist), 0, 1) ** 2
        glare_mask = np.stack([glare_mask] * 3, axis=-1)
        
        glare_color = np.array([200, 220, 255])  # Warm yellow-white
        image = np.clip(image + glare_mask * glare_color * 0.3, 0, 255).astype(np.uint8)
        
        return image
    
    def apply_dust(self, image: np.ndarray, intensity: float = 0.3) -> np.ndarray:
        """Apply dust/particle effect."""
        h, w = image.shape[:2]
        
        # Create dust particles
        dust_mask = np.random.rand(h, w) > (1 - intensity * 0.02)
        dust_mask = cv2.GaussianBlur(dust_mask.astype(np.float32), (5, 5), 0)
        
        # Add brownish dust haze
        haze = np.ones_like(image) * np.array([180, 190, 200])  # Brown-ish
        dust_alpha = dust_mask[:, :, np.newaxis] * intensity * 0.5
        
        image = np.clip(image * (1 - dust_alpha) + haze * dust_alpha, 0, 255).astype(np.uint8)
        
        return image
    
    def apply_monsoon_rain(self, image: np.ndarray, intensity: float = 0.5) -> np.ndarray:
        """Apply monsoon rain effect with rain streaks."""
        h, w = image.shape[:2]
        
        # Create rain streaks
        rain_layer = np.zeros((h, w), dtype=np.uint8)
        
        n_drops = int(h * w * intensity * 0.001)
        for _ in range(n_drops):
            x = np.random.randint(0, w)
            y = np.random.randint(0, h)
            length = np.random.randint(10, 30)
            cv2.line(rain_layer, (x, y), (x + 2, y + length), 255, 1)
        
        rain_layer = cv2.GaussianBlur(rain_layer, (3, 3), 0)
        
        # Apply rain layer
        rain_3d = np.stack([rain_layer] * 3, axis=-1) / 255.0 * 0.5
        image = np.clip(image + rain_3d * 100, 0, 255).astype(np.uint8)
        
        # Reduce saturation (rainy look)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] *= 0.7
        hsv[:, :, 2] *= 0.85
        image = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        
        return image
    
    def apply_humidity_haze(self, image: np.ndarray, intensity: float = 0.2) -> np.ndarray:
        """Apply high humidity haze effect."""
        # Add slight white haze
        haze = np.ones_like(image) * 255
        image = cv2.addWeighted(image, 1 - intensity, haze, intensity, 0)
        
        return image


# =============================================================================
# HARD NEGATIVE GENERATOR
# =============================================================================

class HardNegativeGenerator:
    """
    Generates hard negative samples for improved model training.
    Creates images without signs but with similar visual patterns.
    """
    
    def __init__(self, seed: Optional[int] = None):
        if seed is not None:
            np.random.seed(seed)
            
    def generate_no_sign_image(self, background: np.ndarray) -> np.ndarray:
        """Generate background-only image (no sign)."""
        return background.copy()
    
    def generate_partially_occluded(
        self,
        composite: np.ndarray,
        bbox: Tuple[int, int, int, int],
        occlusion_ratio: float = 0.5
    ) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
        """
        Generate partially occluded sign image.
        
        Args:
            composite: Composite image with sign
            bbox: Bounding box of sign
            occlusion_ratio: How much of sign to occlude (0-1)
            
        Returns:
            (occluded_image, adjusted_bbox)
        """
        x, y, w, h = bbox
        output = composite.copy()
        
        # Create occlusion (simulate tree branch, pole, etc.)
        occlude_type = np.random.choice(['branch', 'pole', 'fade'])
        
        if occlude_type == 'branch':
            # Diagonal branch-like occlusion
            pts = np.array([
                [x, y],
                [x + int(w * occlusion_ratio), y],
                [x + int(w * (occlusion_ratio + 0.1)), y + h],
                [x, y + h]
            ], np.int32)
            
            # Green-ish color for vegetation
            color = (30 + np.random.randint(30), 100 + np.random.randint(50), 20)
            cv2.fillPoly(output, [pts], color)
            
        elif occlude_type == 'pole':
            # Vertical pole occlusion
            pole_x = x + int(w * 0.3)
            pole_w = max(5, int(w * 0.1))
            cv2.rectangle(output, (pole_x, y - 20), (pole_x + pole_w, y + h + 20), 
                         (60, 60, 60), -1)
            
        else:  # Fade/dirty
            # Dirty/faded effect on part of sign
            roi = output[y:y+h, x:x+int(w*occlusion_ratio)]
            roi = cv2.addWeighted(roi, 0.3, np.ones_like(roi) * 128, 0.7, 0)
            output[y:y+h, x:x+int(w*occlusion_ratio)] = roi
        
        return output, bbox


# =============================================================================
# SIGN POLE RENDERER (NEW ENHANCEMENT)
# =============================================================================

class SignPoleRenderer:
    """
    Renders realistic sign poles/posts below traffic signs.
    Makes signs look mounted instead of floating.
    """
    
    def __init__(self):
        # Pole colors (gray metal, rusty, black)
        self.pole_colors = [
            (70, 70, 70),    # Dark gray
            (90, 90, 90),    # Medium gray
            (50, 50, 50),    # Darker
            (60, 70, 80),    # Slightly blue-gray
            (80, 70, 60),    # Slightly rusty
        ]
    
    def add_pole(
        self, 
        image: np.ndarray, 
        sign_bbox: Tuple[int, int, int, int],
        extend_to_bottom: bool = True
    ) -> np.ndarray:
        """
        Add a realistic pole below the sign.
        """
        x, y, w, h = sign_bbox
        img_h, img_w = image.shape[:2]
        
        # Pole properties
        pole_width = max(4, int(w * 0.08))  # Proportional to sign width
        pole_center_x = x + w // 2
        
        # Ensure pole width is reasonable
        pole_width = min(pole_width, 40)
        
        pole_start_y = y + h  # Start just below sign
        pole_end_y = img_h if extend_to_bottom else min(img_h, y + h + int(h * 3))
        
        # Random pole color with noise
        base_color = self.pole_colors[np.random.randint(len(self.pole_colors))]
        
        # Draw pole with cylindrical gradient (dark-light-dark)
        for i in range(pole_width):
            # Calculate brightness factor for cylindrical effect
            # Edges are darker (0.6x), center is brighter (1.0x)
            rel_x = i / max(1, pole_width)
            factor = 0.6 + 0.4 * np.sin(rel_x * np.pi)
            
            # Add random texture noise
            noise = np.random.randint(-20, 20)
            
            # Calculate color for this strip
            col = [np.clip(c * factor + noise, 0, 255) for c in base_color]
            
            # Draw vertical line strip
            x_pos = int(pole_center_x - pole_width/2 + i)
            cv2.line(image, (x_pos, pole_start_y), (x_pos, pole_end_y), col, 1)
            
        return image


# =============================================================================
# WHITE BACKGROUND FIXER (NEW ENHANCEMENT)
# =============================================================================

class WhiteBackgroundFixer:
    """
    Handles PNG templates that have white backgrounds instead of transparency.
    Creates proper alpha masks for clean compositing.
    """
    
    def __init__(self, white_threshold: int = 245):
        self.white_threshold = white_threshold
    
    def fix_transparency(self, image: np.ndarray) -> np.ndarray:
        """
        Convert white background to transparent using flood fill.
        Preserves white pixels inside the sign.
        
        Args:
            image: Input image (BGR or BGRA)
            
        Returns:
            BGRA image with white converted to transparent
        """
        # Ensure 4 channels
        if image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
        
        # Get current alpha
        bgr = image[:, :, :3]
        alpha = image[:, :, 3]
        
        # Detect white pixels (high R, G, B values)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        
        # Also check if all channels are similar (white/gray)
        b, g, r = cv2.split(bgr)
        color_variance = np.abs(r.astype(int) - g.astype(int)) + \
                        np.abs(g.astype(int) - b.astype(int))
        
        # White pixels: high brightness AND low color variance
        white_mask = (gray > self.white_threshold) & (color_variance < 30)
        
        # Convert to uint8 for floodFill (255 = white candidate, 0 = non-white)
        flood_mask = white_mask.astype(np.uint8) * 255
        h, w = flood_mask.shape
        
        # Mask for floodFill needs to be 2 pixels larger
        mask = np.zeros((h+2, w+2), np.uint8)
        
        # Flood fill from all 4 corners if they are white
        # We fill with value 128 to distinguish "background white" from "content white"
        corners = [(0, 0), (w-1, 0), (0, h-1), (w-1, h-1)]
        
        background_found = False
        for cx, cy in corners:
            if flood_mask[cy, cx] == 255:
                cv2.floodFill(flood_mask, mask, (cx, cy), 128, loDiff=0, upDiff=0)
                background_found = True
        
        if background_found:
            # Only remove the flooded areas (value 128)
            alpha[flood_mask == 128] = 0
            
            # Anti-alias the edges of the transparency
            bg_mask = (flood_mask == 128).astype(np.uint8)
            kernel = np.ones((3, 3), np.uint8)
            edge_mask = cv2.dilate(bg_mask, kernel) - bg_mask
            
            # Make edges semi-transparent
            alpha[edge_mask > 0] = np.minimum(alpha[edge_mask > 0], 128)
        else:
            # Fallback for when corners aren't white? 
            # If no corners are white, maybe the image doesn't feature a white background box.
            # In that case, do nothing (preserve sign content).
            pass
        
        image[:, :, 3] = alpha
        
        return image
    
    def fill_transparent_holes(self, image: np.ndarray) -> np.ndarray:
        """
        Fill transparent holes inside the sign with white.
        Common issue with downloaded PNGs where white areas are transparent.
        """
        if image.shape[2] != 4:
            return image
            
        # Get alpha channel
        alpha = image[:, :, 3]
        
        # Find contours of non-transparent regions
        contours, _ = cv2.findContours(alpha, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return image
            
        # Create a white background mask
        mask = np.zeros_like(alpha)
        
        # Draw all external contours filled with white
        cv2.drawContours(mask, contours, -1, 255, -1)
        
        # Create a white background image
        white_bg = np.full_like(image, 255)
        white_bg[:, :, 3] = mask  # Use the contour mask as alpha
        
        # Composite original image on top of white background
        # This keeps the original sign content/colors but fills holes with white
        
        # Convert to float for blending
        fg = image.astype(float) / 255.0
        bg = white_bg.astype(float) / 255.0
        
        # Alpha blending
        alpha_fg = fg[:, :, 3:4]
        alpha_bg = bg[:, :, 3:4]
        
        # Result alpha is union of both
        alpha_out = np.maximum(alpha_fg, alpha_bg)
        
        # Result color
        # Where fg has alpha, keep fg. Where fg is transparent but bg has alpha (hole), use white.
        color_out = fg[:, :, :3] * alpha_fg + bg[:, :, :3] * alpha_bg * (1 - alpha_fg)
        
        # Normalize by output alpha to avoid darkening (pre-multiplied alpha handling)
        color_out = color_out / (alpha_out + 1e-6)
        
        # Reconstruct result
        result = np.zeros_like(image)
        result[:, :, :3] = np.clip(color_out * 255, 0, 255).astype(np.uint8)
        result[:, :, 3] = np.clip(alpha_out * 255, 0, 255).astype(np.uint8).squeeze()
        
        return result

    def ensure_alpha(self, image: np.ndarray) -> np.ndarray:
        """
        Ensure image has proper alpha channel.
        Handles both transparent PNGs and white-background PNGs.
        """
        if image.shape[2] == 3:
            # No alpha channel - likely white background
            return self.fix_transparency(image)
        
        elif image.shape[2] == 4:
            # Check if alpha is all 255 (no transparency)
            alpha = image[:, :, 3]
            mean_alpha = np.mean(alpha)
            
            if mean_alpha > 250:  # Mostly opaque
                # Likely white background, try to fix
                return self.fix_transparency(image)
            else:
                # Has transparency, but might have holes (transparent sign face)
                # Apply fill hole fix
                return self.fill_transparent_holes(image)
            
        return image


# =============================================================================
# COLOR MATCHER (NEW ENHANCEMENT)
# =============================================================================

class ColorMatcher:
    """
    Matches sign colors to background lighting conditions.
    Makes compositing more realistic.
    """
    
    def __init__(self):
        pass
    
    def get_background_color_temperature(self, background: np.ndarray) -> str:
        """
        Analyze background to determine color temperature.
        
        Returns:
            'warm', 'neutral', or 'cool'
        """
        # Convert to HSV for analysis
        hsv = cv2.cvtColor(background, cv2.COLOR_BGR2HSV)
        
        # Get average hue and saturation
        avg_hue = np.mean(hsv[:, :, 0])
        avg_sat = np.mean(hsv[:, :, 1])
        avg_val = np.mean(hsv[:, :, 2])
        
        # Warm: orangish hues (10-30) with decent saturation
        # Cool: bluish hues (100-130) 
        # Neutral: low saturation or mid hues
        
        if avg_sat < 30:
            return 'neutral'  # Low saturation = neutral/gray
        elif avg_hue < 30 or avg_hue > 160:
            return 'warm'  # Red/orange/yellow hues
        elif 90 < avg_hue < 140:
            return 'cool'  # Blue hues
        else:
            return 'neutral'
    
    def match_sign_to_background(
        self, 
        sign: np.ndarray, 
        background: np.ndarray,
        intensity: float = 0.3
    ) -> np.ndarray:
        """
        Adjust sign colors to match background lighting.
        
        Args:
            sign: Sign image (BGRA)
            background: Background image
            intensity: How much to adjust (0-1)
            
        Returns:
            Color-adjusted sign
        """
        temp = self.get_background_color_temperature(background)
        
        # Get sign BGR (preserve alpha)
        if sign.shape[2] == 4:
            alpha = sign[:, :, 3:4]
            bgr = sign[:, :, :3]
        else:
            alpha = None
            bgr = sign
        
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
        
        if temp == 'warm':
            # Add warm tint (shift toward orange, increase brightness slightly)
            hsv[:, :, 0] = np.clip(hsv[:, :, 0] - 5 * intensity, 0, 179)  # Shift hue toward red
            hsv[:, :, 2] = np.clip(hsv[:, :, 2] * (1 + 0.1 * intensity), 0, 255)  # Slightly brighter
            
        elif temp == 'cool':
            # Add cool tint (desaturate slightly, blue shift)
            hsv[:, :, 1] = hsv[:, :, 1] * (1 - 0.2 * intensity)  # Desaturate
            hsv[:, :, 2] = hsv[:, :, 2] * (1 - 0.1 * intensity)  # Slightly darker
            
        # else neutral - no change
        
        bgr_adjusted = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        
        if alpha is not None:
            return np.concatenate([bgr_adjusted, alpha], axis=2)
        return bgr_adjusted
    
    def apply_ambient_lighting(
        self,
        sign: np.ndarray,
        background: np.ndarray
    ) -> np.ndarray:
        """
        Apply ambient lighting from background to sign.
        Uses average background color to tint the sign slightly.
        """
        # Sample background color
        bg_sample = background[
            background.shape[0]//3:2*background.shape[0]//3,
            background.shape[1]//3:2*background.shape[1]//3
        ]
        avg_color = np.mean(bg_sample, axis=(0, 1))
        
        # Very subtle tint (5-10%)
        if sign.shape[2] == 4:
            alpha = sign[:, :, 3:4]
            bgr = sign[:, :, :3].astype(np.float32)
        else:
            alpha = None
            bgr = sign.astype(np.float32)
        
        # Blend very slightly with ambient color
        tint_strength = 0.05
        tinted = bgr * (1 - tint_strength) + avg_color * tint_strength
        tinted = np.clip(tinted, 0, 255).astype(np.uint8)
        
        if alpha is not None:
            return np.concatenate([tinted, alpha], axis=2)
        return tinted


# =============================================================================
# SYNTHETIC AUGMENTOR
# =============================================================================

class SyntheticAugmentor:
    """
    Applies heavy augmentation to synthetic images.
    Includes motion blur, camera shake, and standard augmentations.
    """
    
    def __init__(self, seed: Optional[int] = None):
        self.seed = seed
        if seed is not None:
            np.random.seed(seed)
            
        self.sl_effects = SriLankaEffects(seed)
        
        if ALBUMENTATIONS_AVAILABLE:
            self.heavy_augment = A.Compose([
                A.OneOf([
                    A.MotionBlur(blur_limit=(3, 9), p=1.0),
                    A.GaussianBlur(blur_limit=(3, 7), p=1.0),
                ], p=0.4),
                A.RandomBrightnessContrast(
                    brightness_limit=0.3,
                    contrast_limit=0.3,
                    p=0.6
                ),
                A.HueSaturationValue(
                    hue_shift_limit=10,
                    sat_shift_limit=20,
                    val_shift_limit=20,
                    p=0.4
                ),
                A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
                A.ImageCompression(quality_lower=50, quality_upper=95, p=0.3),
            ], p=1.0)
        else:
            self.heavy_augment = None
            
    def apply_foreground_occlusion(self, image: np.ndarray) -> np.ndarray:
        """
        Add random foreground occlusion (branches, leaves) blocking part of the image.
        Uses the whole image to place occluders, potentially covering the sign.
        """
        h, w = image.shape[:2]
        output = image.copy()
        
        # Occlusion type: vegetation or wire/pole
        if np.random.rand() < 0.7: # Mostly vegetation
            # Vegetation blobs
            num_blobs = np.random.randint(3, 8)
            # Pick a cluster center
            cluster_x = np.random.randint(0, w)
            cluster_y = np.random.randint(0, h)
            
            for _ in range(num_blobs):
                # Random position near cluster
                blob_x = cluster_x + np.random.randint(-100, 100)
                blob_y = cluster_y + np.random.randint(-100, 100)
                
                # Random color (dark green/brown/shadowy)
                color = (
                    np.random.randint(20, 60), # B
                    np.random.randint(40, 100), # G
                    np.random.randint(20, 60)   # R
                )
                
                # Draw blob (circle)
                radius = np.random.randint(20, 80)
                cv2.circle(output, (blob_x, blob_y), radius, color, -1)
                
                # Soften edges? Hard to do efficiently without alpha blending layers.
                # Just keeping it hard edged for speed or maybe naive blur later.
                
        else:
            # Wire/Pole
            start_pt = (np.random.randint(0, w), np.random.randint(0, h))
            end_pt = (np.random.randint(0, w), np.random.randint(0, h))
            thickness = np.random.randint(2, 8)
            color = (30, 30, 30)
            cv2.line(output, start_pt, end_pt, color, thickness)
            
        return output
            
    def apply_motion_blur(self, image: np.ndarray, strength: int = 5) -> np.ndarray:
        """Apply motion blur effect."""
        # Create motion blur kernel
        kernel = np.zeros((strength, strength))
        kernel[int((strength - 1) / 2), :] = np.ones(strength)
        kernel = kernel / strength
        
        return cv2.filter2D(image, -1, kernel)
    
    def apply_camera_shake(self, image: np.ndarray) -> np.ndarray:
        """Apply camera shake effect."""
        h, w = image.shape[:2]
        
        # Random shift
        dx = np.random.randint(-3, 4)
        dy = np.random.randint(-2, 3)
        
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        shifted = cv2.warpAffine(image, M, (w, h))
        
        # Blend with original for ghost effect
        return cv2.addWeighted(image, 0.7, shifted, 0.3, 0)
    
    def augment(
        self,
        image: np.ndarray,
        weather: WeatherCondition = WeatherCondition.CLEAR
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Apply augmentations to image.
        
        Returns:
            (augmented_image, list_of_applied_augmentations)
        """
        applied = []
        
        # Apply weather effects
        if weather == WeatherCondition.RAIN:
            image = self.sl_effects.apply_monsoon_rain(image)
            applied.append("monsoon_rain")
        elif weather == WeatherCondition.FOG:
            image = self.sl_effects.apply_humidity_haze(image, 0.3)
            applied.append("fog_haze")
        elif weather == WeatherCondition.DUST:
            image = self.sl_effects.apply_dust(image)
            applied.append("dust")
        elif weather == WeatherCondition.HARSH_SUN:
            image = self.sl_effects.apply_tropical_sun(image)
            applied.append("tropical_sun")
            
        # Apply motion effects randomly
        if np.random.rand() < 0.2:
            image = self.apply_motion_blur(image, np.random.randint(3, 7))
            applied.append("motion_blur")
            
        if np.random.rand() < 0.15:
            image = self.apply_camera_shake(image)
            applied.append("camera_shake")
            
        # Apply foreground occlusion (20% chance)
        if np.random.rand() < 0.2:
            image = self.apply_foreground_occlusion(image)
            applied.append("foreground_occlusion")
            
        # Apply albumentations if available
        if self.heavy_augment is not None:
            result = self.heavy_augment(image=image)
            image = result['image']
            applied.append("albumentations_heavy")
                
        return image, applied
        



# =============================================================================
# MAIN SYNTHETIC DATA GENERATOR
# =============================================================================

class SyntheticDataGenerator:
    """
    Main orchestrator for synthetic data generation.
    Combines all components for end-to-end generation.
    """
    
    def __init__(
        self,
        template_dir: Path,
        background_dir: Path,
        output_dir: Path,
        seed: int = 42
    ):
        self.template_dir = Path(template_dir)
        self.background_dir = Path(background_dir)
        self.output_dir = Path(output_dir)
        self.seed = seed
        
        # Set seeds for reproducibility
        np.random.seed(seed)
        random.seed(seed)
        
        # Initialize components
        self.compositor = ContextAwareCompositor(seed)
        self.realism_filter = RealismFilter()
        self.augmentor = SyntheticAugmentor(seed)
        self.hard_neg_gen = HardNegativeGenerator(seed)
        
        # NEW: Enhancement components
        self.pole_renderer = SignPoleRenderer()
        self.bg_fixer = WhiteBackgroundFixer()
        self.color_matcher = ColorMatcher()
        
        # Create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "images").mkdir(exist_ok=True)
        (self.output_dir / "hard_negatives").mkdir(exist_ok=True)
        
        # Metadata storage
        self.metadata: List[ImageMetadata] = []
        
    def load_templates(self) -> Dict[str, List[Path]]:
        """Load all template images organized by category."""
        templates = {}
        
        for category_key, category_info in SIGN_CONTEXT_MAPPING.items():
            folder_path = self.template_dir / "online_collected" / category_info.folder
            if folder_path.exists():
                images = list(folder_path.glob("*.png")) + list(folder_path.glob("*.jpg"))
                if images:
                    templates[category_key] = images
                    logger.info(f"Loaded {len(images)} templates from {category_key}")
                    
        # Also load from direct subfolders
        # Exclude road_markings since they are painted on roads, not mounted signs
        excluded_folders = ["regulatory_signs", "directional_informative_signs", "road_markings"]
        for subfolder in (self.template_dir / "online_collected").iterdir():
            if subfolder.is_dir() and subfolder.name not in excluded_folders:
                images = list(subfolder.glob("*.png")) + list(subfolder.glob("*.jpg"))
                if images and subfolder.name not in templates:
                    templates[subfolder.name] = images
                    logger.info(f"Loaded {len(images)} templates from {subfolder.name}")
                    
        return templates
    
    def load_backgrounds(self) -> Dict[str, List[Path]]:
        """Load all background images organized by condition."""
        backgrounds = {}
        
        # First check for images directly in the background_dir
        direct_images = list(self.background_dir.glob("*.png")) + \
                       list(self.background_dir.glob("*.jpg")) + \
                       list(self.background_dir.glob("*.jpeg"))
        
        if direct_images:
            # Images found directly in folder (e.g., when using --bg-category)
            folder_name = self.background_dir.name
            backgrounds[folder_name] = direct_images
            logger.info(f"Loaded {len(direct_images)} backgrounds from {folder_name}")
        
        # Also check subdirectories
        for subfolder in self.background_dir.iterdir():
            if subfolder.is_dir():
                images = list(subfolder.glob("*.png")) + \
                         list(subfolder.glob("*.jpg")) + \
                         list(subfolder.glob("*.jpeg"))
                if images:
                    backgrounds[subfolder.name] = images
                    logger.info(f"Loaded {len(images)} backgrounds from {subfolder.name}")
                    
        return backgrounds
    
    def _get_weather_from_folder(self, folder_name: str) -> WeatherCondition:
        """Map background folder to weather condition."""
        mapping = {
            "rain": WeatherCondition.RAIN,
            "fog_mist": WeatherCondition.FOG,
            "harsh_conditions": WeatherCondition.HARSH_SUN,
            "urban_day": WeatherCondition.CLEAR,
            "highway": WeatherCondition.CLEAR,
            "night": WeatherCondition.CLEAR,
            "golden_hour": WeatherCondition.CLEAR,
            "coastal": WeatherCondition.CLEAR,
            "rural": WeatherCondition.DUST,
            "school_residential": WeatherCondition.CLEAR,
        }
        return mapping.get(folder_name, WeatherCondition.CLEAR)
    
    def generate_single(
        self,
        template_path: Path,
        background_path: Path,
        category: str,
        variation_idx: int
    ) -> Optional[Tuple[np.ndarray, ImageMetadata]]:
        """
        Generate a single synthetic image.
        
        Returns:
            (image, metadata) or None if rejected
        """
        # Load images
        template = cv2.imread(str(template_path), cv2.IMREAD_UNCHANGED)
        background = cv2.imread(str(background_path))
        
        if template is None or background is None:
            logger.warning(f"Failed to load: {template_path} or {background_path}")
            return None
            
        # NEW: Fix white background and ensure alpha channel
        template = self.bg_fixer.ensure_alpha(template)
        
        # NEW: Match sign colors to background lighting
        template = self.color_matcher.match_sign_to_background(template, background)
            
        # Get depth/scale manager
        bg_h, bg_w = background.shape[:2]
        depth_manager = DepthScaleManager(bg_h, bg_w)
        
        # Get valid placement zones
        x_zones, y_min_global, y_max_global = depth_manager.get_valid_placement_zone()
        
        # Optimize Y selection to satisfy realism filter (sign size)
        img_aspect = bg_w / bg_h
        sign_aspect = template.shape[1] / template.shape[0]
        
        # scale = sqrt(ratio * img_aspect / sign_aspect)
        min_req = np.sqrt(self.realism_filter.min_sign_size * img_aspect / sign_aspect)
        max_req = np.sqrt(self.realism_filter.max_sign_size * img_aspect / sign_aspect)
        
        y_opt_min, y_opt_max = depth_manager.get_y_range_for_scale(min_req, max_req)
        
        # Intersect with global constraints
        y_min = max(y_min_global, y_opt_min)
        y_max = min(y_max_global, y_opt_max)
        
        if y_min >= y_max:
             # Fallback
             y_min = y_min_global
             y_max = y_max_global
        
        # Random position
        x_zone = random.choice(x_zones)
        x = np.random.randint(x_zone[0], x_zone[1])
        y = np.random.randint(y_min, y_max)
        
        # Get depth-appropriate scale
        scale = depth_manager.get_scale_for_position(y)
        
        # Add some random variation to scale
        scale *= np.random.uniform(0.8, 1.2)
        scale = np.clip(scale, 0.05, 0.3)
        
        # Composite
        # Increased perspective probability (was 0.7)
        apply_perspective = np.random.rand() < 0.9
        light_direction = np.random.uniform(0, np.pi)
        
        composite, bbox = self.compositor.composite_sign(
            background,
            template,
            (x, y),
            scale,
            apply_perspective,
            light_direction
        )
        
        # NEW: Add sign pole for realism (70% chance)
        # NEW: Add sign pole for realism (Always add unless configured otherwise)
        if True:  # Changed from 0.7 probability to always based on user feedback
            composite = self.pole_renderer.add_pole(composite, bbox)
        
        # Check realism
        is_valid, reason = self.realism_filter.check_composition(
            composite, bbox, background
        )
        
        if not is_valid:
            logger.debug(f"Rejected: {reason}")
            return None
            
        # Get weather condition from background folder
        bg_folder = background_path.parent.name
        weather = self._get_weather_from_folder(bg_folder)
        
        # Apply augmentations
        augmented, augs_applied = self.augmentor.augment(composite, weather)
        
        # Create metadata
        metadata = ImageMetadata(
            source_template=str(template_path.name),
            background_file=str(background_path.name),
            weather_condition=weather.value,
            time_of_day=self._infer_time_of_day(bg_folder),
            road_context=bg_folder,
            sign_position=(x, y),
            sign_scale=scale,
            perspective_applied=apply_perspective,
            augmentations=augs_applied,
            bbox=bbox,
            random_seed=self.seed + variation_idx,
            generation_timestamp=datetime.now().isoformat(),
            is_hard_negative=False
        )
        
        return augmented, metadata
    
    def _infer_time_of_day(self, folder_name: str) -> str:
        """Infer time of day from folder name."""
        if "night" in folder_name:
            return TimeOfDay.NIGHT.value
        elif "golden" in folder_name:
            return TimeOfDay.GOLDEN_HOUR.value
        elif "fog" in folder_name or "mist" in folder_name:
            return TimeOfDay.MORNING.value
        else:
            return TimeOfDay.NOON.value
    
    def generate_dataset(
        self,
        n_per_template: int = 100,
        include_hard_negatives: bool = True,
        hard_negative_ratio: float = 0.1
    ) -> Dict[str, Any]:
        """
        Generate complete synthetic dataset.
        
        Args:
            n_per_template: Number of variations per template
            include_hard_negatives: Generate hard negative samples
            hard_negative_ratio: Ratio of hard negatives to positives
            
        Returns:
            Statistics dictionary
        """
        templates = self.load_templates()
        backgrounds = self.load_backgrounds()
        
        if not templates:
            raise ValueError(f"No templates found in {self.template_dir}")
        if not backgrounds:
            raise ValueError(f"No backgrounds found in {self.background_dir}")
            
        total_templates = sum(len(t) for t in templates.values())
        total_backgrounds = sum(len(b) for b in backgrounds.values())
        
        logger.info(f"Starting generation: {total_templates} templates × {n_per_template} variations")
        logger.info(f"Using {total_backgrounds} background images")
        
        stats = {
            "total_generated": 0,
            "rejected": 0,
            "hard_negatives": 0,
            "per_category": {}
        }
        
        all_backgrounds = []
        for folder, paths in backgrounds.items():
            for p in paths:
                all_backgrounds.append((folder, p))
        
        image_idx = 0
        
        for category, template_list in templates.items():
            category_count = 0
            
            for template_path in template_list:
                for var_idx in range(n_per_template):
                    # Select random background
                    bg_folder, bg_path = random.choice(all_backgrounds)
                    
                    # Generate
                    result = self.generate_single(
                        template_path, bg_path, category, image_idx + var_idx
                    )
                    
                    if result is None:
                        stats["rejected"] += 1
                        continue
                        
                    image, metadata = result
                    
                    # Save image
                    output_name = f"syn_{image_idx:06d}.jpg"
                    cv2.imwrite(
                        str(self.output_dir / "images" / output_name),
                        image,
                        [cv2.IMWRITE_JPEG_QUALITY, 95]
                    )
                    
                    metadata_dict = asdict(metadata)
                    metadata_dict["output_file"] = output_name
                    # Convert tuples to lists for JSON serialization
                    if "sign_position" in metadata_dict:
                        metadata_dict["sign_position"] = list(metadata_dict["sign_position"])
                    if "bbox" in metadata_dict:
                        metadata_dict["bbox"] = list(metadata_dict["bbox"])
                    self.metadata.append(metadata_dict)
                    
                    stats["total_generated"] += 1
                    category_count += 1
                    image_idx += 1
                    
                    if image_idx % 100 == 0:
                        logger.info(f"Generated {image_idx} images...")
            
            stats["per_category"][category] = category_count
        
        # Generate hard negatives
        if include_hard_negatives:
            n_hard_neg = int(stats["total_generated"] * hard_negative_ratio)
            logger.info(f"Generating {n_hard_neg} hard negatives...")
            
            for i in range(n_hard_neg):
                bg_folder, bg_path = random.choice(all_backgrounds)
                background = cv2.imread(str(bg_path))
                
                if background is not None:
                    # Just use background with some augmentation
                    weather = self._get_weather_from_folder(bg_folder)
                    augmented, _ = self.augmentor.augment(background, weather)
                    
                    output_name = f"hardneg_{i:05d}.jpg"
                    cv2.imwrite(
                        str(self.output_dir / "hard_negatives" / output_name),
                        augmented,
                        [cv2.IMWRITE_JPEG_QUALITY, 95]
                    )
                    stats["hard_negatives"] += 1
        
        # Save metadata
        metadata_path = self.output_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=2, cls=NumpyEncoder)
        logger.info(f"Saved metadata to {metadata_path}")
        
        # Save statistics
        stats_path = self.output_dir / "generation_stats.json"
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2, cls=NumpyEncoder)
        logger.info(f"Saved statistics to {stats_path}")
        
        return stats


# =============================================================================
# CLI INTERFACE
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate synthetic traffic sign data")
    parser.add_argument("--templates", type=str, required=True, 
                       help="Path to template images directory")
    parser.add_argument("--backgrounds", type=str, required=True,
                       help="Path to background images directory")
    parser.add_argument("--output", type=str, required=True,
                       help="Output directory for generated data")
    parser.add_argument("--n-per-template", type=int, default=100,
                       help="Number of variations per template")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility")
    parser.add_argument("--no-hard-negatives", action="store_true",
                       help="Disable hard negative generation")
    
    args = parser.parse_args()
    
    generator = SyntheticDataGenerator(
        template_dir=Path(args.templates),
        background_dir=Path(args.backgrounds),
        output_dir=Path(args.output),
        seed=args.seed
    )
    
    stats = generator.generate_dataset(
        n_per_template=args.n_per_template,
        include_hard_negatives=not args.no_hard_negatives
    )
    
    print("\n=== Generation Complete ===")
    print(f"Total images: {stats['total_generated']}")
    print(f"Rejected: {stats['rejected']}")
    print(f"Hard negatives: {stats['hard_negatives']}")
    print("\nPer category:")
    for cat, count in stats['per_category'].items():
        print(f"  {cat}: {count}")
