#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Shape-Based Model Matching Implementation with OpenCV
----------------------------------------------------
This is an implementation inspired by HALCON's Shape-Based Model matching algorithm
Based on gradient orientation extraction and template matching.

Requirements:
- OpenCV
- NumPy
- SciPy (for subpixel refinement)
"""

import cv2
import numpy as np
from scipy.optimize import least_squares
import time
import math
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon


class ShapeBasedMatching:
    """
    A Shape-Based Model matching implementation inspired by HALCON's algorithm
    Uses gradient orientation for matching, making it robust to lighting changes
    """
    
    def __init__(self):
        # Parameters for template creation
        self.min_contrast = 5         # Minimum gradient magnitude to consider as edge
        self.num_features = 200        # Number of feature points to use
        self.pyramid_levels = 1        # Number of scale pyramid levels
        self.angle_step = 10           # Angle step for multiple templates (degrees)
        self.scale_step = 0.1          # Scale step for multiple templates
        
        # Parameters for matching
        self.min_score = 0.2          # Minimum matching score (0.0 to 1.0)
        self.greediness = 0.6          # Balance between accurate and fast matching (0.0 to 1.0)
        self.edge_filter_size = 5      # Edge filter kernel size
        self.max_overlap = 0.6         # Maximum allowed overlap between matches
        self.pyramid_scale = 2.0       # Scale factor between pyramid levels
        
        # Templates
        self.templates = []            # List of template information
        self.angle_range = None        # Angle search range [min_angle, max_angle]
        self.scale_range = None        # Scale search range [min_scale, max_scale]

        self.pyramid_levels = 3  # 默认金字塔层级
        self.lowest_level_to_use = 1  # 追踪到的最低层级
        self.adaptive_lowest_level = True  # 启用自适应最低层级选择
        self.pyramid_scale = 2.0  # 金字塔缩放因子
        self.greediness = 0.9  # 贪婪度参数(0-安全但慢，1-快但可能漏检)
        
    def create_template(self, img, mask=None):
        """
        Create a template from an image
        
        Args:
            img: Template image (grayscale)
            mask: Optional binary mask to specify region of interest
            
        Returns:
            template_id: ID of the created template
        """
        print("[INFO] 开始创建模板...")
        if len(img.shape) > 2:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Create empty mask if not provided
        if mask is None:
            mask = np.ones(img.shape, dtype=np.uint8) * 255
            
        # Preprocess the image
        print("[INFO] 预处理模板图像...")
        img_smooth = cv2.GaussianBlur(img, (3, 3), 0)
        
        # Extract gradient information
        print("[INFO] 提取梯度信息...")
        dx = cv2.Sobel(img_smooth, cv2.CV_32F, 1, 0, ksize=3)
        dy = cv2.Sobel(img_smooth, cv2.CV_32F, 0, 1, ksize=3)
        
        # Calculate gradient magnitude and orientation
        magnitude = np.sqrt(dx**2 + dy**2)
        orientation = np.arctan2(dy, dx) * (180 / np.pi) % 360  # 0-360 degrees
        
        # Apply mask and min contrast threshold
        edges = (magnitude > self.min_contrast) & (mask > 0)
        print(f"[INFO] 找到 {np.sum(edges)} 个边缘点")
        
        # Store edge points for visualization
        edge_points = np.where(edges)
        edge_coords = list(zip(edge_points[1], edge_points[0]))  # Convert to (x,y) format
        
        # Select feature points
        print("[INFO] 选择特征点...")
        feature_points = self._select_feature_points(edges, magnitude, orientation)
        print(f"[INFO] 选择了 {len(feature_points)} 个特征点")
        
        # Create template object
        template = {
            'features': feature_points,
            'width': img.shape[1],
            'height': img.shape[0],
            'center_x': img.shape[1] // 2,
            'center_y': img.shape[0] // 2,
            'angle': 0,
            'scale': 1.0,
            'edge_points': edge_coords,  # Store edge points for visualization
            'original_image': img.copy() if len(img.shape) == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Store original image
        }
        
        # Add template to collection
        template_id = len(self.templates)
        self.templates.append(template)
        print("[INFO] 模板创建完成")
        
        # Display template with edge points and feature points
        self.visualize_template(template_id)
        
        return template_id
    
    def set_angle_range(self, min_angle, max_angle):
        """
        Set the angle search range in degrees
        
        Args:
            min_angle: Minimum angle in degrees (angle_start in HALCON)
            max_angle: Maximum angle in degrees (angle_start + angle_extent in HALCON)
        """
        self.angle_range = [min_angle, max_angle]
        print(f"[INFO] 设置角度范围: {min_angle} 到 {max_angle} 度")
        return self
    
    def set_scale_range(self, min_scale, max_scale):
        """
        Set the scale search range
        
        Args:
            min_scale: Minimum scale factor (scale_min in HALCON)
            max_scale: Maximum scale factor (scale_max in HALCON)
        """
        self.scale_range = [min_scale, max_scale]
        print(f"[INFO] 设置缩放范围: {min_scale} 到 {max_scale}")
        return self

    def set_min_score(self, min_score, max_clutter_score=None):
        """
        Set the minimum score threshold for matches
        
        Args:
            min_score: Minimum score threshold (0.0 to 1.0)
            max_clutter_score: Optional maximum clutter score threshold (0.0 to 1.0)
        """
        self.min_score = min_score
        self.max_clutter_score = max_clutter_score
        print(f"[INFO] 设置最小匹配分数: {min_score}")
        if max_clutter_score is not None:
            print(f"[INFO] 设置最大杂波分数: {max_clutter_score}")
        return self

    def set_max_overlap(self, max_overlap):
        """
        Set the maximum allowed overlap between matches
        
        Args:
            max_overlap: Maximum overlap ratio (0.0 to 1.0)
        """
        self.max_overlap = max(0.0, min(1.0, max_overlap))
        print(f"[INFO] 设置最大重叠比例: {self.max_overlap}")
        return self

    def set_greediness(self, greediness):
        """
        Set the greediness parameter for search heuristic
        
        Args:
            greediness: Greediness value (0.0 for safe but slow, 1.0 for fast but potentially missing matches)
        """
        self.greediness = max(0.0, min(1.0, greediness))
        print(f"[INFO] 设置贪婪度参数: {self.greediness}")
        return self

    def set_subpixel_mode(self, mode="interpolation", max_deformation=0):
        """
        Set the subpixel refinement mode
        
        Args:
            mode: Subpixel refinement mode:
                  "none" - No subpixel refinement
                  "interpolation" - Fast interpolation-based refinement
                  "least_squares" - Accurate least-squares refinement
                  "least_squares_high" - High accuracy least-squares refinement
                  "least_squares_very_high" - Very high accuracy least-squares refinement
            max_deformation: Maximum allowed deformation in pixels (0-32)
        """
        valid_modes = ["none", "interpolation", "least_squares", 
                      "least_squares_high", "least_squares_very_high"]
        
        if mode not in valid_modes:
            print(f"[WARNING] 无效的亚像素模式 '{mode}'，使用默认值 'interpolation'")
            mode = "interpolation"
        
        max_deformation = max(0, min(32, int(max_deformation)))
        
        self.subpixel_mode = mode
        self.max_deformation = max_deformation
        
        print(f"[INFO] 设置亚像素精度模式: {mode}")
        if max_deformation > 0:
            print(f"[INFO] 设置最大变形量: {max_deformation} 像素")
        return self

    def _build_image_pyramid(self, image, edges, magnitude, orientation):
        """
        Build image pyramid for multi-scale matching
        
        Args:
            image: Input image
            edges: Edge image
            magnitude: Gradient magnitude
            orientation: Gradient orientation
            
        Returns:
            pyramid: List of (image, edges, magnitude, orientation) tuples
                    from fine to coarse (original image is at index 0)
        """
        pyramid = [(image.copy(), edges.copy(), magnitude.copy(), orientation.copy())]
        
        current_image = image.copy()
        current_edges = edges.copy()
        current_magnitude = magnitude.copy()
        current_orientation = orientation.copy()
        
        for level in range(1, self.pyramid_levels):
            # Calculate new dimensions
            new_height = int(current_image.shape[0] / self.pyramid_scale)
            new_width = int(current_image.shape[1] / self.pyramid_scale)
            
            # Stop if image becomes too small
            if new_height < 30 or new_width < 30:
                break
            
            # Resize image
            resized_image = cv2.resize(current_image, (new_width, new_height), interpolation=cv2.INTER_AREA)
            
            # Option 1: Resize existing gradients (faster but less accurate)
            resized_edges = cv2.resize(current_edges, (new_width, new_height), interpolation=cv2.INTER_AREA)
            resized_magnitude = cv2.resize(current_magnitude, (new_width, new_height), interpolation=cv2.INTER_AREA)
            resized_orientation = cv2.resize(current_orientation, (new_width, new_height), interpolation=cv2.INTER_NEAREST)
            
            # Add to pyramid
            pyramid.append((resized_image, resized_edges, resized_magnitude, resized_orientation))
            
            # Update current level
            current_image = resized_image
            current_edges = resized_edges
            current_magnitude = resized_magnitude
            current_orientation = resized_orientation
        
        return pyramid

    def find_model_pyramid(self, img, num_matches=1, timeout_ms=None):
        """
        Find the template in the target image using image pyramid for faster processing
        Similar to HALCON's find_scaled_shape_model
        
        Args:
            img: Target image
            num_matches: Maximum number of matches to find (0 for all matches above min_score)
            timeout_ms: Optional timeout in milliseconds
            
        Returns:
            matches: List of match objects with position, angle, scale and score
        """
        print("[INFO] 开始使用图像金字塔在目标图像中查找模板...")
        if len(img.shape) > 2:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        original_height, original_width = img.shape
        print(f"[INFO] 目标图像尺寸: {original_width}x{original_height}")
        
        # Start timer if timeout specified
        start_time = time.time() if timeout_ms else None
        
        # Preprocess the image
        print("[INFO] 尝试增强的预处理...")
        # 标准化图像亮度
        img_norm = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
        # 应用自适应直方图均衡化
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img_enhanced = clahe.apply(img_norm)
        # 使用增强后的图像
        img_smooth = cv2.GaussianBlur(img_enhanced, (5, 5), 0)
        
        # Extract gradient information
        print("[INFO] 提取梯度信息...")
        dx = cv2.Sobel(img_smooth, cv2.CV_32F, 1, 0, ksize=3)
        dy = cv2.Sobel(img_smooth, cv2.CV_32F, 0, 1, ksize=3)
        
        # Calculate gradient magnitude and orientation
        magnitude = np.sqrt(dx**2 + dy**2)
        orientation = np.arctan2(dy, dx) * (180 / np.pi) % 360  # 0-360 degrees
        
        # Apply min contrast threshold
        edges = (magnitude > self.min_contrast).astype(np.uint8) * 255
        print(f"[INFO] 找到 {np.sum(edges > 0)} 个边缘点")
        
        # Build image pyramid
        print(f"[INFO] 构建图像金字塔 (最大 {self.pyramid_levels} 层)...")
        pyramid = self._build_image_pyramid(img, edges, magnitude, orientation)
        print(f"[INFO] 图像金字塔构建完成，共 {len(pyramid)} 层")
        
        for i, (layer_img, _, _, _) in enumerate(pyramid):
            print(f"[INFO] 金字塔层级 {i}: 大小 {layer_img.shape[1]}x{layer_img.shape[0]}")
        
        # Initialize matches list
        all_matches = []
        
        # Match each template
        for template_id, template in enumerate(self.templates):
            print(f"[INFO] 开始匹配模板 {template_id}...")
            
            # Get template features
            features = template['features']
            
            # Default angles and scales if not set
            angles = [0] if not self.angle_range else np.arange(self.angle_range[0], self.angle_range[1], self.angle_step)
            scales = [1.0] if not self.scale_range else np.arange(self.scale_range[0], self.scale_range[1] + self.scale_step, self.scale_step)
            
            print(f"[INFO] 将搜索 {len(angles)} 个角度变化和 {len(scales)} 个缩放变化")
            
            # Determine pyramid levels to use
            start_level = len(pyramid) - 1
            lowest_level_to_use = self.lowest_level_to_use
            
            # Handle adaptive lowest level (similar to HALCON's negative num_levels)
            adaptive_mode = lowest_level_to_use < 0
            if adaptive_mode:
                lowest_level_to_use = abs(lowest_level_to_use)
                print(f"[INFO] 启用自适应金字塔层级模式，初始最低层级: {lowest_level_to_use}")
            
            # Ensure valid pyramid levels
            lowest_level_to_use = max(0, min(start_level, lowest_level_to_use))
            
            # Keep track of promising locations
            candidates = []
            
            # Coarse search at the coarsest level
            print(f"[INFO] 在金字塔顶层 (层级 {start_level}) 进行粗略搜索...")
            coarse_img, coarse_edges, coarse_magnitude, coarse_orientation = pyramid[start_level]
            h, w = coarse_img.shape
            
            # Scale for this pyramid level
            level_scale = 1.0 / (self.pyramid_scale ** start_level)
            
            # Scale margins based on template size
            template_width = template['width'] * level_scale
            template_height = template['height'] * level_scale
            x_margin = int(template_width / 2)
            y_margin = int(template_height / 2)
            
            # Larger step size for coarse search - adjusted by greediness
            # Lower greediness = smaller step size (safer search)
            coarse_step = max(1, int(min(w, h) / 100 * (0.5 + self.greediness * 0.5)))
            
            # Lower score threshold for candidates - adjusted by greediness
            # Lower greediness = lower threshold (more candidates)
            min_candidate_score = self.min_score * (0.3 + self.greediness * 0.2)
            
            # For each angle and scale combination
            for angle in angles:
                for scale in scales:
                    # Check timeout
                    if timeout_ms and (time.time() - start_time) * 1000 > timeout_ms:
                        print(f"[INFO] 超时 {timeout_ms}ms，返回当前匹配结果")
                        return all_matches
                    
                    # Scale and rotate features
                    scaled_features = []
                    for fx, fy, fangle in features:
                        # Scale to match pyramid level
                        scaled_x = fx * level_scale * scale
                        scaled_y = fy * level_scale * scale
                        # Rotate feature
                        rotated_x = scaled_x * math.cos(math.radians(angle)) - scaled_y * math.sin(math.radians(angle))
                        rotated_y = scaled_x * math.sin(math.radians(angle)) + scaled_y * math.cos(math.radians(angle))
                        # Adjust feature angle
                        adjusted_angle = (fangle + angle) % 360
                        scaled_features.append([rotated_x, rotated_y, adjusted_angle])
                    
                    print(f"[INFO] 粗略搜索: 角度={angle:.1f}度, 缩放={scale:.2f}")
                    
                    # Coarse search using larger step size
                    for y in range(y_margin, h - y_margin, coarse_step):
                        for x in range(x_margin, w - x_margin, coarse_step):
                            # Compute match score
                            score, _ = self._compute_match_score_optimized(
                                scaled_features, coarse_magnitude, coarse_orientation, x, y
                            )
                            
                            # Store promising candidates
                            if score > min_candidate_score:
                                # Convert to original image coordinates
                                orig_x = int(x * (self.pyramid_scale ** start_level))
                                orig_y = int(y * (self.pyramid_scale ** start_level))
                                candidates.append({
                                    'x': orig_x,
                                    'y': orig_y,
                                    'angle': angle,
                                    'scale': scale,
                                    'score': score,
                                    'level': start_level
                                })
            
            # Sort candidates by score
            candidates.sort(key=lambda c: c['score'], reverse=True)
            
            # Limit number of candidates to process based on greediness
            # Lower greediness = more candidates (safer search)
            max_candidates = min(int(20 + (1 - self.greediness) * 30), len(candidates))
            candidates = candidates[:max_candidates]
            
            print(f"[INFO] 在粗略搜索中找到 {len(candidates)} 个候选匹配点")
            
            # Refine candidates through pyramid levels
            for level in range(start_level - 1, lowest_level_to_use - 1, -1):
                refined_candidates = []
                
                # Get images for this level
                level_img, level_edges, level_magnitude, level_orientation = pyramid[level]
                h, w = level_img.shape
                
                # Scale for this pyramid level
                level_scale = 1.0 / (self.pyramid_scale ** level)
                
                # Scale margins based on template size
                template_width = template['width'] * level_scale
                template_height = template['height'] * level_scale
                x_margin = int(template_width / 2)
                y_margin = int(template_height / 2)
                
                # Search parameters for this level - adjusted by greediness
                search_radius = int(3 * self.pyramid_scale * (1 + (1 - self.greediness) * 0.5))
                search_step = max(1, int(3 / self.pyramid_scale * (0.5 + self.greediness * 0.5)))
                
                # Adjust min score based on pyramid level and greediness
                level_min_score = self.min_score * (0.5 + 0.5 * (1 - level / start_level))
                
                print(f"[INFO] 在金字塔层级 {level} 精细化搜索，共 {len(candidates)} 个候选点...")
                
                # Process each candidate
                for cand_idx, candidate in enumerate(candidates):
                    # Convert candidate position to current level coordinates
                    level_x = int(candidate['x'] / (self.pyramid_scale ** level))
                    level_y = int(candidate['y'] / (self.pyramid_scale ** level))
                    
                    # Get angle and scale from candidate
                    angle = candidate['angle']
                    scale = candidate['scale']
                    
                    # For the final level, we'll use smaller step sizes for angles and scales
                    angle_step_fine = self.angle_step / 2 if level == lowest_level_to_use else self.angle_step
                    angle_range = [max(min(angles), angle - angle_step_fine), 
                                  min(max(angles), angle + angle_step_fine)]
                    
                    scale_step_fine = self.scale_step / 2 if level == lowest_level_to_use else self.scale_step
                    scale_range = [max(min(scales), scale - scale_step_fine), 
                                  min(max(scales), scale + scale_step_fine)]
                    
                    # Refine angles and scales
                    best_score = 0
                    best_x = level_x
                    best_y = level_y
                    best_angle = angle
                    best_scale = scale
                    
                    for fine_angle in np.arange(angle_range[0], angle_range[1] + angle_step_fine, angle_step_fine):
                        for fine_scale in np.arange(scale_range[0], scale_range[1] + scale_step_fine, scale_step_fine):
                            # Scale and rotate features
                            scaled_features = []
                            for fx, fy, fangle in features:
                                # Scale to match pyramid level
                                scaled_x = fx * level_scale * fine_scale
                                scaled_y = fy * level_scale * fine_scale
                                # Rotate feature
                                rotated_x = scaled_x * math.cos(math.radians(fine_angle)) - scaled_y * math.sin(math.radians(fine_angle))
                                rotated_y = scaled_x * math.sin(math.radians(fine_angle)) + scaled_y * math.cos(math.radians(fine_angle))
                                # Adjust feature angle
                                adjusted_angle = (fangle + fine_angle) % 360
                                scaled_features.append([rotated_x, rotated_y, adjusted_angle])
                            
                            # Define search area around candidate
                            x_min = max(x_margin, level_x - search_radius)
                            x_max = min(w - x_margin, level_x + search_radius)
                            y_min = max(y_margin, level_y - search_radius)
                            y_max = min(h - y_margin, level_y + search_radius)
                            
                            # Search within this area
                            for y in range(y_min, y_max, search_step):
                                for x in range(x_min, x_max, search_step):
                                    # Compute match score
                                    score, _ = self._compute_match_score_optimized(
                                        scaled_features, level_magnitude, level_orientation, x, y
                                    )
                                    
                                    # Update best match
                                    if score > best_score:
                                        best_score = score
                                        best_x = x
                                        best_y = y
                                        best_angle = fine_angle
                                        best_scale = fine_scale
                    
                    # Check if refined match is good enough
                    if best_score > level_min_score:
                        # Convert to original image coordinates
                        orig_x = int(best_x * (self.pyramid_scale ** level))
                        orig_y = int(best_y * (self.pyramid_scale ** level))
                        
                        # For the final level, add to matches
                        if level == lowest_level_to_use:
                            match = {
                                'x': orig_x,
                                'y': orig_y,
                                'angle': best_angle,
                                'scale': best_scale,
                                'score': best_score,
                                'template_id': template_id
                            }
                            
                            # Refine to subpixel accuracy if enabled
                            if hasattr(self, 'subpixel_mode') and self.subpixel_mode != "none":
                                try:
                                    refined_match = self._refine_position_subpixel(match, magnitude, orientation, features)
                                    all_matches.append(refined_match)
                                    print(f"[INFO] 找到匹配 #{len(all_matches)}: 分数={refined_match['score']:.4f}, 位置=({refined_match['x']:.2f}, {refined_match['y']:.2f}), 角度={refined_match['angle']:.2f}度, 缩放={refined_match['scale']:.2f}")
                                except Exception as e:
                                    print(f"[WARNING] 亚像素优化失败: {e}")
                                    all_matches.append(match)
                                    print(f"[INFO] 找到匹配 #{len(all_matches)}: 分数={match['score']:.4f}, 位置=({match['x']}, {match['y']}), 角度={match['angle']:.2f}度, 缩放={match['scale']:.2f}")
                            else:
                                all_matches.append(match)
                                print(f"[INFO] 找到匹配 #{len(all_matches)}: 分数={match['score']:.4f}, 位置=({match['x']}, {match['y']}), 角度={match['angle']:.2f}度, 缩放={match['scale']:.2f}")
                        else:
                            # Add to candidates for next level
                            refined_candidates.append({
                                'x': orig_x,
                                'y': orig_y,
                                'angle': best_angle,
                                'scale': best_scale,
                                'score': best_score,
                                'level': level
                            })
                
                # Update candidates for next level
                if level > 0:
                    candidates = refined_candidates
                    # Sort by score and limit number
                    candidates.sort(key=lambda c: c['score'], reverse=True)
                    max_candidates = min(10, len(candidates))
                    candidates = candidates[:max_candidates]
                    print(f"[INFO] 层级 {level} 精细化后保留 {len(candidates)} 个候选点")
        
        # Filter overlapping matches
        if all_matches:
            filtered_matches = self._filter_overlapping_matches(all_matches)
            print(f"[INFO] 过滤重叠匹配后剩余 {len(filtered_matches)} 个匹配")
            return filtered_matches[:num_matches]
        else:
            print("[INFO] 未找到匹配")
            return []

    def _select_feature_points(self, edges, magnitude, orientation):
        """
        Select a subset of feature points with high gradient magnitude
        and uniform spatial distribution
        
        Args:
            edges: Binary edge mask
            magnitude: Gradient magnitude
            orientation: Gradient orientation
            
        Returns:
            features: List of feature points [x, y, orientation]
        """
        # Get edge points
        y_indices, x_indices = np.where(edges)
        
        if len(y_indices) == 0:
            return []
        
        # Sort points by magnitude in descending order
        mags = magnitude[y_indices, x_indices]
        sorted_indices = np.argsort(-mags)
        
        x_indices = x_indices[sorted_indices]
        y_indices = y_indices[sorted_indices]
        
        # Initialize distance map to ensure even distribution
        min_dist = int(np.sqrt(edges.shape[0] * edges.shape[1] / self.num_features) * 0.7)
        dist_map = np.zeros(edges.shape, dtype=np.float32)
        
        # Select feature points
        features = []
        for i in range(len(y_indices)):
            x, y = x_indices[i], y_indices[i]
            
            if dist_map[y, x] == 0:  # If point is not too close to existing points
                # Add to features
                features.append([x, y, orientation[y, x]])
                
                # Update distance map
                y_min = max(0, y - min_dist)
                y_max = min(edges.shape[0], y + min_dist + 1)
                x_min = max(0, x - min_dist)
                x_max = min(edges.shape[1], x + min_dist + 1)
                
                for yy in range(y_min, y_max):
                    for xx in range(x_min, x_max):
                        dist_sq = (xx - x)**2 + (yy - y)**2
                        if dist_sq <= min_dist**2:
                            dist_map[yy, xx] = 1
                            
            if len(features) >= self.num_features:
                break
                
        return features
    
    def _transform_features(self, features, angle, scale):
        """
        Transform template features by rotation and scaling
        
        Args:
            features: Template features
            angle: Rotation angle in degrees
            scale: Scale factor
            
        Returns:
            transformed_features: Transformed template features
        """
        transformed_features = []
        
        angle_rad = angle * np.pi / 180
        cos_angle = np.cos(angle_rad)
        sin_angle = np.sin(angle_rad)
        
        for feature in features:
            x, y, orientation = feature
            
            # Rotate and scale coordinates
            x_new = scale * (x * cos_angle - y * sin_angle)
            y_new = scale * (x * sin_angle + y * cos_angle)
            
            # Rotate orientation
            orientation_new = (orientation + angle) % 360
            
            transformed_features.append([x_new, y_new, orientation_new])
            
        return transformed_features
        
    def _match_template(self, features, magnitude, orientation, width, height, center_x, center_y):
        """
        Match the template features in the target image using pyramid approach for faster searching
        
        Args:
            features: Template features
            magnitude: Target image gradient magnitude
            orientation: Target image gradient orientation
            width: Template width
            height: Template height
            center_x: Template center X coordinate
            center_y: Template center Y coordinate
            
        Returns:
            match: Match object with position and score
        """
        h, w = magnitude.shape
        
        # Define search boundaries
        x_margin = int(width / 2)
        y_margin = int(height / 2)
        
        print(f"[INFO] 搜索范围: x={x_margin} 到 {w-x_margin}, y={y_margin} 到 {h-y_margin}")
        
        # Initialize best match
        best_score = -1
        best_x = -1
        best_y = -1
        
        # Use two-level pyramid approach: first search with a larger step, then refine
        # First level: Coarse search with large step - adjusted based on image size
        coarse_step = max(15, int(min(w, h) / 200))  # Dynamic step size based on image dimensions
        print(f"[INFO] 第一阶段: 使用步长 {coarse_step} 进行粗略搜索...")
        
        # 为跟踪进度添加计数器
        total_coarse_positions = ((h - 2*y_margin) // coarse_step + 1) * ((w - 2*x_margin) // coarse_step + 1)
        print(f"[INFO] 粗略搜索需要检查 {total_coarse_positions} 个位置")
        position_count = 0
        last_percent = -1
        
        coarse_candidates = []  # Store promising positions for refinement
        
        # For each possible position in coarse search
        for y in range(y_margin, h - y_margin, coarse_step):
            for x in range(x_margin, w - x_margin, coarse_step):
                position_count += 1
                
                # 每搜索10%的位置打印一次进度
                current_percent = (position_count * 100) // total_coarse_positions
                if current_percent % 10 == 0 and current_percent != last_percent:
                    print(f"[INFO] 粗略搜索进度: {current_percent}% ({position_count}/{total_coarse_positions})")
                    last_percent = current_percent
                
                # Compute score for current position
                score, num_matched = self._compute_match_score_optimized(features, magnitude, orientation, x, y)
                
                # Store promising candidates for refinement (score > threshold)
                if score > self.min_score * 0.5:  # Lower threshold for candidates
                    coarse_candidates.append((x, y, score))
                
                # Update best match
                if score > best_score:
                    best_score = score
                    best_x = x
                    best_y = y
                    
                    # Early termination if we found an excellent match
                    if score > 0.8:  # Lowered from 0.9
                        print(f"[INFO] 找到高质量匹配，提前终止搜索: score={score:.3f}, 位置=({best_x}, {best_y})")
                        coarse_candidates.append((x, y, score))
                        break
        
            # Early termination using greediness parameter
            if best_score > 0.8:  # Lowered from 0.9
                break
        
        # Second level: Refine search near promising locations
        print(f"[INFO] 第二阶段: 精细搜索，在 {len(coarse_candidates)} 个候选位置附近使用较小步长...")
        fine_step = 2  # Smaller step for refined search
        
        # If no promising candidates were found, add the best match anyway
        if not coarse_candidates and best_score > 0:
            coarse_candidates.append((best_x, best_y, best_score))
        
        # For each promising candidate
        for candidate_idx, (cx, cy, score) in enumerate(sorted(coarse_candidates, key=lambda x: x[2], reverse=True)[:20]):  # Increased from 10 to 20
            print(f"[INFO] 精细搜索候选位置 {candidate_idx+1}/{min(20, len(coarse_candidates))}: 初始score={score:.3f}, 位置=({cx}, {cy})")
            
            # Define search radius for refinement
            refine_radius = coarse_step + 4  # Increased from 2 to 4
            
            # Search around the candidate position
            for y in range(max(y_margin, cy - refine_radius), min(h - y_margin, cy + refine_radius + 1), fine_step):
                for x in range(max(x_margin, cx - refine_radius), min(w - x_margin, cx + refine_radius + 1), fine_step):
                    # Compute score for current position
                    score, num_matched = self._compute_match_score_optimized(features, magnitude, orientation, x, y)
                    
                    # Update best match
                    if score > best_score:
                        best_score = score
                        best_x = x
                        best_y = y
                        
                        # Early termination if we found an excellent match
                        if score > 0.8:  # Lowered from 0.9
                            print(f"[INFO] 在精细搜索中找到高质量匹配: score={score:.3f}, 位置=({best_x}, {best_y})")
                            break
            
            # Early termination using greediness parameter
            if score > 0.8:  # Lowered from 0.9
                break
        
        print(f"[INFO] 搜索完成，最佳匹配: score={best_score:.3f}, 位置=({best_x}, {best_y})")
        
        # Return the best match
        if best_score >= self.min_score:
            return {
                'x': best_x,
                'y': best_y,
                'score': best_score
            }
        else:
            return None
        
    def _compute_match_score_optimized(self, features, magnitude, orientation, center_x, center_y):
        """
        Compute match score between template features and target image at given position
        Optimized version with faster NumPy operations
        
        Args:
            features: Template features
            magnitude: Target image gradient magnitude
            orientation: Target image gradient orientation
            center_x: Position X coordinate
            center_y: Position Y coordinate
            
        Returns:
            score: Match score (0.0 to 1.0)
            num_matched: Number of matched features
        """
        h, w = magnitude.shape
        num_features = len(features)
        
        if num_features == 0:
            return 0, 0
        
        # 将所有特征点转换为目标图像中的坐标 (用NumPy矢量化操作)
        feature_array = np.array(features)
        x_coords = np.clip(center_x + feature_array[:, 0], 0, w-1).astype(np.int32)
        y_coords = np.clip(center_y + feature_array[:, 1], 0, h-1).astype(np.int32)
        feature_angles = feature_array[:, 2]
        
        # 一次性获取所有梯度值
        target_mags = magnitude[y_coords, x_coords]
        target_angles = orientation[y_coords, x_coords]
        
        # 只考虑梯度大于阈值的点
        valid_indices = target_mags > self.min_contrast
        valid_count = np.sum(valid_indices)
        
        if valid_count == 0:
            return 0, 0
        
        # 计算有效点的角度差异 (处理角度循环)
        feature_angles_valid = feature_angles[valid_indices]
        target_angles_valid = target_angles[valid_indices]
        
        # 高效计算角度差异 (考虑循环)
        angle_diffs = np.minimum(
            np.abs(target_angles_valid - feature_angles_valid), 
            360 - np.abs(target_angles_valid - feature_angles_valid)
        )
        
        # 转换为相似度分数 (0到1)
        angle_similarities = np.maximum(0, 1.0 - angle_diffs / 90.0)
        num_matched = np.sum(angle_diffs < 45)
        
        # 计算平均相似度分数 (相对于总特征点数)
        if len(angle_similarities) > 0:
            score = np.sum(angle_similarities) / num_features
        else:
            score = 0.0
        
        return score, num_matched
    
    def _refine_position(self, match, magnitude, orientation, features):
        """
        Refine match position to subpixel accuracy
        
        Args:
            match: Initial match result
            magnitude: Target image gradient magnitude
            orientation: Target image gradient orientation
            features: Template features
            
        Returns:
            refined_match: Match with refined position
        """
        print(f"[INFO] 开始亚像素优化，初始位置: ({match['x']}, {match['y']})")
        # Initial parameters: x, y, and potential angle and scale
        x0 = [match['x'], match['y']]
        
        # Define the objective function to minimize
        def objective_func(params):
            x, y = params
            residuals = []
            
            for feature in features:
                fx, fy, fangle = feature
                
                # Calculate feature position in target image
                img_x = int(x + fx)
                img_y = int(y + fy)
                
                h, w = magnitude.shape
                
                # Check if position is within bounds
                if 0 <= img_x < w and 0 <= img_y < h:
                    # Get target gradient magnitude and orientation
                    target_mag = magnitude[img_y, img_x]
                    target_angle = orientation[img_y, img_x]
                    
                    # If target has strong gradient
                    if target_mag > self.min_contrast:
                        # Compute orientation difference (considering circularity)
                        angle_diff = min(abs(target_angle - fangle), 360 - abs(target_angle - fangle))
                        
                        # Add to residuals
                        residuals.append(angle_diff)
            
            return np.array(residuals)
        
        # Run least squares optimization
        try:
            print(f"[INFO] 运行最小二乘优化...")
            result = least_squares(objective_func, x0, method='lm', ftol=1e-4, xtol=1e-4)
            
            # Create refined match
            refined_match = match.copy()
            refined_match['x'] = result.x[0]
            refined_match['y'] = result.x[1]
            
            print(f"[INFO] 优化后位置: ({refined_match['x']:.2f}, {refined_match['y']:.2f})")
        except Exception as e:
            print(f"[ERROR] 优化失败: {str(e)}")
            # If optimization fails, return original match
            return match
            
    def _filter_overlapping_matches(self, matches):
        """
        Filter out overlapping matches, keeping the one with the highest score
        
        Args:
            matches: List of match objects
            
        Returns:
            filtered_matches: List of non-overlapping matches
        """
        matches = sorted(matches, key=lambda m: m['score'], reverse=True)
        filtered_matches = []
        
        for match in matches:
            # Check overlap with already accepted matches
            overlapping = False
            for accepted_match in filtered_matches:
                # Calculate overlap ratio
                overlap = self._calculate_overlap(match, accepted_match)
                
                if overlap > self.max_overlap:
                    overlapping = True
                    break
            
            if not overlapping:
                filtered_matches.append(match)
                
        return filtered_matches
    
    def _calculate_overlap(self, match1, match2):
        """
        Calculate the overlap ratio between two matches
        
        Args:
            match1: First match
            match2: Second match
            
        Returns:
            overlap: Overlap ratio (0.0 to 1.0)
        """
        # For simplicity, use circle approximation for now
        # A more accurate calculation would use actual template shape
        template1 = self.templates[match1['template_id']]
        template2 = self.templates[match2['template_id']]
        
        r1 = max(template1['width'], template1['height']) / 2 * match1['scale']
        r2 = max(template2['width'], template2['height']) / 2 * match2['scale']
        
        dx = match1['x'] - match2['x']
        dy = match1['y'] - match2['y']
        distance = math.sqrt(dx*dx + dy*dy)
        
        # If circles are not overlapping
        if distance >= r1 + r2:
            return 0.0
        
        # If one circle is inside the other
        if distance <= abs(r1 - r2):
            return 1.0
        
        # Partial overlap
        area1 = math.pi * r1 * r1
        area2 = math.pi * r2 * r2
        
        # Calculate overlap area using circle segment formula
        d = distance
        a = (r1*r1 - r2*r2 + d*d) / (2*d)
        b = d - a
        
        h1 = r1 - a
        h2 = r2 - b
        
        if h1 <= 0:
            return 0.0
        if h2 <= 0:
            return 0.0
        
        area_segment1 = r1*r1 * math.acos((r1-h1)/r1) - (r1-h1) * math.sqrt(2*r1*h1 - h1*h1)
        area_segment2 = r2*r2 * math.acos((r2-h2)/r2) - (r2-h2) * math.sqrt(2*r2*h2 - h2*h2)
        
        overlap_area = area_segment1 + area_segment2
        smaller_area = min(area1, area2)
        
        return overlap_area / smaller_area

    def _scale_matches_to_original(self, matches, downsample_factor):
        """Helper method to scale matches back to original image size"""
        print("[INFO] 将匹配坐标转换回原始图像比例...")
        for match in matches:
            match['x'] /= downsample_factor
            match['y'] /= downsample_factor
            # Note: scale doesn't need adjustment as it's relative

    def visualize_template(self, template_id):
        """
        Visualize template with edge points and feature points
        
        Args:
            template_id: ID of the template to visualize
        """
        template = self.templates[template_id]
        
        # Create figure with 2 subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
        
        # Original image with edge points
        ax1.imshow(template['original_image'], cmap='gray')
        
        # Plot edge points
        edge_x = [x for x, y in template['edge_points']]
        edge_y = [y for x, y in template['edge_points']]
        
        # If there are many edge points, sample them to avoid plotting too many
        max_points_to_plot = 5000
        if len(edge_x) > max_points_to_plot:
            sample_rate = len(edge_x) // max_points_to_plot
            edge_x = edge_x[::sample_rate]
            edge_y = edge_y[::sample_rate]
        
        ax1.scatter(edge_x, edge_y, s=1, color='blue', alpha=0.3)
        ax1.set_title(f'Template Edge Points ({len(template["edge_points"])} points)')
        
        # Original image with feature points
        ax2.imshow(template['original_image'], cmap='gray')
        
        # Plot feature points
        feature_x = [x for x, y, _ in template['features']]
        feature_y = [y for x, y, _ in template['features']]
        
        # Plot feature points with markers showing orientation
        for x, y, angle in template['features']:
            ax2.plot(x, y, 'ro', markersize=3)
            
            # Draw a line indicating the orientation
            line_length = 5
            dx = line_length * np.cos(angle * np.pi / 180)
            dy = line_length * np.sin(angle * np.pi / 180)
            ax2.plot([x, x + dx], [y, y + dy], 'r-', linewidth=0.5)
        
        ax2.set_title(f'Template Feature Points ({len(template["features"])} points)')
        
        # Remove ticks
        ax1.set_xticks([])
        ax1.set_yticks([])
        ax2.set_xticks([])
        ax2.set_yticks([])
        
        plt.tight_layout()
        
        # Add a descriptive title
        plt.suptitle(f'Template Visualization (ID: {template_id}, Size: {template["width"]}x{template["height"]})',
                    fontsize=12, y=0.98)
        
        # Option to save the visualization
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        save_path = f"template_visualization_{timestamp}.png"
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"[INFO] 模板可视化已保存至: {save_path}")
        
        # Show the plot
        plt.show()

    def _refine_position_subpixel(self, match, magnitude, orientation, features):
        """
        Refine match position to subpixel accuracy using the specified mode
        Similar to HALCON's subpixel parameter in find_scaled_shape_model
        
        Args:
            match: Initial match result
            magnitude: Target image gradient magnitude
            orientation: Target image gradient orientation
            features: Template features
            
        Returns:
            refined_match: Match with refined position
        """
        print(f"[INFO] 开始亚像素优化，初始位置: ({match['x']}, {match['y']})")
        
        # Create a copy of the match to refine
        refined_match = match.copy()
        
        # Simple interpolation mode (fast)
        if not hasattr(self, 'subpixel_mode') or self.subpixel_mode == "none":
            return refined_match
        
        try:
            # Initial parameters: x, y
            x0 = [match['x'], match['y']]
            
            # Define the objective function to minimize
            def objective_func(params):
                x, y = params
                residuals = []
                
                for feature in features:
                    fx, fy, fangle = feature
                    
                    # Apply scale and rotation
                    scaled_fx = fx * match['scale']
                    scaled_fy = fy * match['scale']
                    
                    # Rotate feature
                    angle_rad = math.radians(match['angle'])
                    rotated_fx = scaled_fx * math.cos(angle_rad) - scaled_fy * math.sin(angle_rad)
                    rotated_fy = scaled_fx * math.sin(angle_rad) + scaled_fy * math.cos(angle_rad)
                    
                    # Calculate feature position in target image
                    img_x = x + rotated_fx
                    img_y = y + rotated_fy
                    
                    h, w = magnitude.shape
                    
                    # Check if position is within bounds
                    if 0 <= int(img_x) < w-1 and 0 <= int(img_y) < h-1:
                        # Get target gradient magnitude and orientation at integer position
                        x0, y0 = int(img_x), int(img_y)
                        target_mag = magnitude[y0, x0]
                        target_angle = orientation[y0, x0]
                        
                        # If target has strong gradient
                        if target_mag > self.min_contrast:
                            # Compute orientation difference (considering circularity)
                            adjusted_fangle = (fangle + match['angle']) % 360
                            angle_diff = min(abs(target_angle - adjusted_fangle), 
                                            360 - abs(target_angle - adjusted_fangle))
                            
                            # Add to residuals
                            residuals.append(angle_diff)
            
            # Run least squares optimization
            print(f"[INFO] 运行位置优化...")
            result = least_squares(objective_func, x0, method='lm', ftol=1e-3, xtol=1e-3, max_nfev=50)
            
            # Update refined match
            refined_match['x'] = result.x[0]
            refined_match['y'] = result.x[1]
            
            print(f"[INFO] 优化后位置: ({refined_match['x']:.2f}, {refined_match['y']:.2f})")
            
            # If we want to refine angle and scale too (for least_squares modes)
            if hasattr(self, 'subpixel_mode') and self.subpixel_mode.startswith("least_squares"):
                # Initial parameters: x, y, angle, scale
                x0 = [refined_match['x'], refined_match['y'], match['angle'], match['scale']]
                
                # Define the objective function to minimize
                def objective_func_full(params):
                    x, y, angle, scale = params
                    residuals = []
                    
                    for feature in features:
                        fx, fy, fangle = feature
                        
                        # Apply scale and rotation
                        scaled_fx = fx * scale
                        scaled_fy = fy * scale
                        
                        # Rotate feature
                        angle_rad = math.radians(angle)
                        rotated_fx = scaled_fx * math.cos(angle_rad) - scaled_fy * math.sin(angle_rad)
                        rotated_fy = scaled_fx * math.sin(angle_rad) + scaled_fy * math.cos(angle_rad)
                        
                        # Calculate feature position in target image
                        img_x = x + rotated_fx
                        img_y = y + rotated_fy
                        
                        h, w = magnitude.shape
                        
                        # Check if position is within bounds
                        if 0 <= int(img_x) < w-1 and 0 <= int(img_y) < h-1:
                            # Get target gradient magnitude and orientation
                            x0, y0 = int(img_x), int(img_y)
                            target_mag = magnitude[y0, x0]
                            target_angle = orientation[y0, x0]
                            
                            # If target has strong gradient
                            if target_mag > self.min_contrast:
                                # Compute orientation difference (considering circularity)
                                adjusted_fangle = (fangle + angle) % 360
                                angle_diff = min(abs(target_angle - adjusted_fangle), 
                                                360 - abs(target_angle - adjusted_fangle))
                                
                                # Add to residuals
                                residuals.append(angle_diff)
                
                # Run least squares optimization
                print(f"[INFO] 运行完整参数优化...")
                result = least_squares(objective_func_full, x0, method='lm', ftol=1e-3, xtol=1e-3, max_nfev=100)
                
                # Update refined match
                refined_match['x'] = result.x[0]
                refined_match['y'] = result.x[1]
                refined_match['angle'] = result.x[2]
                refined_match['scale'] = result.x[3]
                
                print(f"[INFO] 优化后位置: ({refined_match['x']:.2f}, {refined_match['y']:.2f}), " + 
                     f"角度: {refined_match['angle']:.2f}, 缩放: {refined_match['scale']:.2f}")
            
            # Handle deformation if enabled
            if hasattr(self, 'max_deformation') and self.max_deformation > 0:
                print(f"[INFO] 变形优化暂未实现，最大变形量设置为: {self.max_deformation}")
                pass  # 将来实现变形优化
            
            return refined_match
            
        except Exception as e:
            print(f"[ERROR] 亚像素优化失败: {str(e)}")
            # If optimization fails, return original match
            return match


def draw_matches(img, matches, model, color=(0, 255, 0), thickness=2):
    """
    Draw matches on the image using OpenCV
    
    Args:
        img: Image to draw on
        matches: List of match objects
        model: ShapeBasedMatching model
        color: Line color
        thickness: Line thickness
    
    Returns:
        img_result: Image with drawn matches
    """
    img_result = img.copy()
    
    for match in matches:
        template = model.templates[match['template_id']]
        
        # Draw match point (top-left corner)
        cv2.circle(img_result, (int(match['x']), int(match['y'])), 5, (0, 0, 255), -1)
        
        # Draw bounding box
        width = template['width'] * match['scale']
        height = template['height'] * match['scale']
        
        # Calculate corners
        angle_rad = match['angle'] * np.pi / 180
        cos_angle = np.cos(angle_rad)
        sin_angle = np.sin(angle_rad)
        
        # Define the four corners from top-left
        corners = [
            (0, 0),                # top-left (match point)
            (width, 0),            # top-right
            (width, height),       # bottom-right
            (0, height)            # bottom-left
        ]
        
        # Rotate corners around the top-left point and translate
        rotated_corners = []
        for x, y in corners:
            # Apply rotation around origin
            x_rot = x * cos_angle - y * sin_angle
            y_rot = x * sin_angle + y * cos_angle
            
            # Translate to match position
            x_final = x_rot + match['x']
            y_final = y_rot + match['y']
            
            rotated_corners.append((int(x_final), int(y_final)))
        
        # Draw the bounding box
        for i in range(4):
            # Draw corner points with different colors for debugging
            corner_colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255)]
            cv2.circle(img_result, rotated_corners[i], 4, corner_colors[i], -1)
            
            # Draw lines connecting corners
            cv2.line(img_result, rotated_corners[i], rotated_corners[(i+1)%4], color, thickness)
        
        # Draw score with background for better visibility
        score_text = f"Score: {match['score']:.2f}"
        text_size = cv2.getTextSize(score_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0]
        text_x = int(match['x'] - text_size[0]/2)
        text_y = int(match['y'] - 10)
        
        # Draw text background
        cv2.rectangle(img_result, 
                     (text_x - 5, text_y - text_size[1] - 5),
                     (text_x + text_size[0] + 5, text_y + 5),
                     color, -1)
        
        # Draw text
        cv2.putText(img_result, score_text, 
                   (text_x, text_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
    
    return img_result


def draw_matches_plt(img, matches, model, color='lime', linewidth=2):
    """
    Draw matches on the image using matplotlib
    
    Args:
        img: Image to draw on
        matches: List of match objects
        model: ShapeBasedMatching model
        color: Line color
        linewidth: Line width
    
    Returns:
        fig, ax: Matplotlib figure and axis objects
    """
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Convert BGR to RGB if needed (OpenCV uses BGR, matplotlib uses RGB)
    if len(img.shape) == 3 and img.shape[2] == 3:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        img_rgb = img
    
    # Display the image
    ax.imshow(img_rgb)
    
    for match in matches:
        template = model.templates[match['template_id']]
        
        # Get template dimensions
        width = template['width'] * match['scale']
        height = template['height'] * match['scale']
        
        # Calculate rotation angle in radians
        angle_rad = match['angle'] * np.pi / 180
        cos_angle = np.cos(angle_rad)
        sin_angle = np.sin(angle_rad)
        
        # Draw match point (this is the top-left corner, not center)
        ax.plot(match['x'], match['y'], 'o', color='red', markersize=8)
        
        # The match coordinates represent the top-left corner in screen coordinates
        # We need to define the full rectangle from this point
        corners = [
            (0, 0),                # top-left (match point)
            (width, 0),            # top-right
            (width, height),       # bottom-right
            (0, height)            # bottom-left
        ]
        
        # Rotate corners around the top-left point and translate
        rotated_corners = []
        for x, y in corners:
            # Apply rotation around origin
            x_rot = x * cos_angle - y * sin_angle
            y_rot = x * sin_angle + y * cos_angle
            
            # Translate to match position
            x_final = x_rot + match['x']
            y_final = y_rot + match['y']
            
            rotated_corners.append((x_final, y_final))
        
        # Draw the bounding box as a polygon
        polygon = Polygon(rotated_corners, closed=True, fill=False, 
                         edgecolor=color, linewidth=linewidth, alpha=0.8)
        ax.add_patch(polygon)
        
        # Draw corner points for debugging with different colors
        corner_colors = ['red', 'green', 'blue', 'yellow']
        for i, (cx, cy) in enumerate(rotated_corners):
            ax.plot(cx, cy, 'o', color=corner_colors[i], markersize=6)
        
        # Draw score with better visibility
        ax.text(match['x'], match['y'] - 15, 
               f"Score: {match['score']:.2f}", 
               color='white', fontsize=10, fontweight='bold',
               bbox=dict(facecolor=color, alpha=0.7),
               horizontalalignment='center')
        
        # Draw match details
        ax.text(match['x'], match['y'] + height + 15,
               f"Angle: {match['angle']:.1f}°, Scale: {match['scale']:.2f}",
               color='white', fontsize=9, fontweight='bold',
               bbox=dict(facecolor='navy', alpha=0.7),
               horizontalalignment='center')
    
    # Add title with match info
    if matches:
        ax.set_title(f"Found {len(matches)} matches", fontsize=14)
    else:
        ax.set_title("No matches found", fontsize=14)
        
    # Remove axis ticks
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Set axis limits to ensure the whole image is visible
    ax.set_xlim(0, img.shape[1])
    ax.set_ylim(img.shape[0], 0)  # Reverse y-axis to match image coordinates
    
    return fig, ax


def resize_for_display(img, max_width=1200, max_height=800):
    """
    Resize an image to fit within specified dimensions while maintaining aspect ratio
    
    Args:
        img: Input image
        max_width: Maximum display width
        max_height: Maximum display height
    
    Returns:
        resized_img: Resized image
    """
    h, w = img.shape[:2]
    
    # Calculate the ratio
    ratio = min(max_width / w, max_height / h)
    
    # Only resize if the image is larger than the maximum dimensions
    if ratio < 1:
        new_w = int(w * ratio)
        new_h = int(h * ratio)
        resized_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        print(f"[INFO] 图像已调整大小以适应显示: {w}x{h} -> {new_w}x{new_h}")
        return resized_img
    
    return img


def _adjust_angle_for_interpolation(angle, reference_angle):
    """
    调整角度以便于插值计算，处理角度环绕问题
    
    Args:
        angle: 需要调整的角度
        reference_angle: 参考角度
        
    Returns:
        adjusted_angle: 调整后的角度
    """
    # 计算角度差异
    diff = angle - reference_angle
    
    # 处理角度环绕
    if diff > 180:
        return angle - 360
    elif diff < -180:
        return angle + 360
    else:
        return angle


# Usage example
if __name__ == "__main__":
    print("程序开始执行...")
    
    # Create model
    model = ShapeBasedMatching()

    # Read template and target images
    print("读取图像文件...")
    template_path = r'C:\Users\goney\Desktop\EmbeddingTest\1\test\tem.png'
    target_path = r'C:\Users\goney\Desktop\EmbeddingTest\1\test\Image_20260206135824717.bmp'
    
    template = cv2.imread(template_path)
    if template is None:
        print(f"错误: 无法读取模板图像: {template_path}")
        exit(1)
    else:
        print(f"成功读取模板图像: {template.shape[1]}x{template.shape[0]}")
        
    target = cv2.imread(target_path)
    if target is None:
        print(f"错误: 无法读取目标图像: {target_path}")
        exit(1)
    else:
        print(f"成功读取目标图像: {target.shape[1]}x{target.shape[0]}")
    
    # Create template
    template_id = model.create_template(template)
    
    # 设置参数
    model.set_min_score(0.2)
    model.set_angle_range(-90, 90)
    model.set_scale_range(0.8, 1.2)
    model.angle_step = 5
    model.scale_step = 0.5
    model.set_greediness(0.5)
    model.set_max_overlap(0.7)
    model.pyramid_levels = 4
    model.lowest_level_to_use = 0
    model.set_subpixel_mode("interpolation")

    # 尝试不同的参数组合
    print("[INFO] 尝试不同参数组合进行匹配...")
    matches = []

    # 尝试组合1: 默认参数
    matches = model.find_model_pyramid(target, num_matches=5, timeout_ms=60000)
    if len(matches) > 0:
        print(f"[INFO] 使用默认参数成功匹配到 {len(matches)} 个结果")
    else:
        # 尝试组合2: 更宽松的参数
        print("[INFO] 默认参数未找到匹配，尝试更宽松的参数...")
        model.set_min_score(0.15)
        model.set_angle_range(-30, 30)
        model.set_scale_range(0.7, 1.3)
        matches = model.find_model_pyramid(target, num_matches=5, timeout_ms=60000)
        
        if len(matches) > 0:
            print(f"[INFO] 使用宽松参数成功匹配到 {len(matches)} 个结果")
        else:
            print("[WARNING] 所有参数组合都未找到匹配，请检查模板和目标图像")

    # Draw matches
    print("绘制匹配结果...")
    
    # Use matplotlib to display the results (has built-in zoom/pan controls)
    fig, ax = draw_matches_plt(target, matches, model)
    
    # Customize the figure
    plt.tight_layout()
    
    # Add a descriptive title with match information
    if matches:
        match_info = " | ".join([f"Match {i+1}: Score={m['score']:.2f}, Angle={m['angle']:.1f}°, Scale={m['scale']:.2f}" 
                                for i, m in enumerate(matches[:3])])
        plt.suptitle(f"Found {len(matches)} matches\n{match_info}", fontsize=10)
    else:
        plt.suptitle("No matches found", fontsize=12)
    
    # Add a note about built-in controls
    plt.figtext(0.5, 0.01, 
               "Tip: Use toolbar for zoom/pan navigation. Press 'h' for help with controls.", 
               ha='center', fontsize=8)
    
    # Show the result with matplotlib's interactive viewer
    print("显示结果...")
    print("使用matplotlib内置控件：放大、缩小、平移等")
    
    # Option to save the result
    save_result = True  # Set to False to disable saving
    if save_result:
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        save_path = f"match_result_{timestamp}.png"
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"结果已保存至: {save_path}")
    
    plt.show()
