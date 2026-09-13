import cv2
import numpy as np
import time
from collections import deque
import math
from scipy import ndimage


class DistanceSmoother:
    """距离平滑器 - 强连续性约束"""

    def __init__(self, max_history=15, max_change_per_frame=1.0):
        self.max_history = max_history
        self.max_change_per_frame = max_change_per_frame  # 每帧最大变化量(米)
        self.distance_history = deque(maxlen=max_history)
        self.velocity_history = deque(maxlen=10)

        # 状态跟踪
        self.last_valid_distance = None
        self.consecutive_failures = 0
        self.stable_count = 0

    def update(self, new_distance, frame_count):
        """更新距离并应用强连续性约束"""
        if new_distance is None:
            self.consecutive_failures += 1
            # 如果连续失败，返回最后一个有效值或None
            if self.consecutive_failures > 5:
                return None
            return self.last_valid_distance

        # 重置失败计数
        self.consecutive_failures = 0

        # 如果没有历史数据，直接使用新值
        if len(self.distance_history) == 0:
            self.distance_history.append(new_distance)
            self.last_valid_distance = new_distance
            return new_distance

        # 获取上一个距离
        last_distance = self.distance_history[-1]

        # 计算距离变化
        distance_change = new_distance - last_distance

        # 强连续性约束：限制每帧最大变化
        if abs(distance_change) > self.max_change_per_frame:
            # 限制变化量在允许范围内
            limited_change = math.copysign(self.max_change_per_frame, distance_change)
            smoothed_distance = last_distance + limited_change

            # 记录异常跳变
            if abs(distance_change) > 5.0:  # 5米以上的跳变被认为是严重异常
                print(f"🚨 帧{frame_count}: 距离跳变过大! {last_distance:.2f}m -> {new_distance:.2f}m "
                      f"(变化: {distance_change:.2f}m), 限制为: {limited_change:.2f}m")
            else:
                print(f"⚠️ 帧{frame_count}: 距离变化较大 {distance_change:.2f}m, 限制为 {limited_change:.2f}m")
        else:
            smoothed_distance = new_distance

        # 应用时间平滑滤波
        if len(self.distance_history) >= 3:
            # 使用加权移动平均，最近帧权重更高
            weights = [0.1, 0.2, 0.7]  # 最近的值权重最高
            recent_distances = list(self.distance_history)[-3:]
            weighted_avg = sum(d * w for d, w in zip(recent_distances, weights))

            # 结合当前值
            smoothed_distance = 0.3 * weighted_avg + 0.7 * smoothed_distance

        # 更新历史记录
        self.distance_history.append(smoothed_distance)
        self.last_valid_distance = smoothed_distance

        # 更新速度历史
        if len(self.distance_history) >= 2:
            velocity = (smoothed_distance - last_distance)  # 每帧变化量
            self.velocity_history.append(velocity)

        return smoothed_distance

    def get_velocity(self):
        """获取当前速度（米/帧）"""
        if len(self.velocity_history) == 0:
            return 0.0
        return np.mean(list(self.velocity_history)[-3:])  # 最近3帧的平均速度

    def get_stability(self):
        """获取稳定性评分"""
        if len(self.distance_history) < 3:
            return 0.0

        recent_distances = list(self.distance_history)[-3:]
        variance = np.var(recent_distances)
        stability = 1.0 / (1.0 + variance * 10)  # 方差越小，稳定性越高
        return stability


class StereoMatcher:
    """立体匹配核心类"""

    def __init__(self, baseline, focal_length):
        self.baseline = baseline
        self.focal_length = focal_length

        self.min_disparity = 0
        self.max_disparity = 160
        self.num_disparities = self.max_disparity - self.min_disparity

        self.setup_matchers()

    def setup_matchers(self):
        """设置匹配器"""
        try:
            self.sgbm_matcher = cv2.StereoSGBM_create(
                minDisparity=self.min_disparity,
                numDisparities=self.num_disparities,
                blockSize=7,
                P1=8 * 3 * 7 ** 2,
                P2=32 * 3 * 7 ** 2,
                disp12MaxDiff=1,
                uniquenessRatio=15,
                speckleWindowSize=100,
                speckleRange=2,
                mode=cv2.STEREO_SGBM_MODE_HH
            )

            self.bm_matcher = cv2.StereoBM_create(
                numDisparities=self.num_disparities,
                blockSize=15
            )

        except Exception as e:
            print(f"匹配器初始化失败: {e}")

    def compute_disparity_sgbm(self, left_img, right_img):
        """使用SGBM计算视差"""
        disparity = self.sgbm_matcher.compute(left_img, right_img)
        return disparity.astype(np.float32) / 16.0

    def compute_disparity_bm(self, left_img, right_img):
        """使用BM计算视差"""
        disparity = self.bm_matcher.compute(left_img, right_img)
        return disparity.astype(np.float32) / 16.0

    def compute_disparity_with_validation(self, left_img, right_img):
        """带验证的视差计算"""
        main_disparity = self.compute_disparity_sgbm(left_img, right_img)
        quality_score = self.evaluate_disparity_quality(main_disparity)

        if quality_score < 0.3:
            backup_disparity = self.compute_disparity_bm(left_img, right_img)
            backup_quality = self.evaluate_disparity_quality(backup_disparity)

            if backup_quality > quality_score:
                return backup_disparity, backup_quality

        return main_disparity, quality_score

    def evaluate_disparity_quality(self, disparity):
        """评估视差图质量"""
        if disparity is None:
            return 0.0

        valid_pixels = np.sum(disparity > 0)
        total_pixels = disparity.size
        valid_ratio = valid_pixels / total_pixels

        if valid_pixels > 0:
            valid_disparities = disparity[disparity > 0]
            disparity_std = np.std(valid_disparities)
            std_score = 1.0 - min(1.0, abs(disparity_std - 20) / 20)
        else:
            std_score = 0.0

        quality = 0.7 * valid_ratio + 0.3 * std_score
        return quality

    def disparity_to_distance(self, disparity):
        """将视差转换为距离"""
        if disparity <= 0:
            return None
        return (self.focal_length * self.baseline) / disparity


class ImageComparator:
    """图像比对分析类"""

    def compare_images_similarity(self, left_img, right_img):
        """比较左右图像的相似度"""
        try:
            hist_similarity = self.histogram_similarity(left_img, right_img)
            feature_similarity = self.feature_matching_similarity(left_img, right_img)

            overall_similarity = 0.6 * hist_similarity + 0.4 * feature_similarity
            return overall_similarity

        except Exception as e:
            print(f"图像相似度计算错误: {e}")
            return 0.5

    def histogram_similarity(self, left_img, right_img):
        """直方图相似度"""
        hist_left = cv2.calcHist([left_img], [0], None, [256], [0, 256])
        hist_right = cv2.calcHist([right_img], [0], None, [256], [0, 256])

        cv2.normalize(hist_left, hist_left, 0, 1, cv2.NORM_MINMAX)
        cv2.normalize(hist_right, hist_right, 0, 1, cv2.NORM_MINMAX)

        similarity = cv2.compareHist(hist_left, hist_right, cv2.HISTCMP_CORREL)
        return max(0, similarity)

    def feature_matching_similarity(self, left_img, right_img):
        """特征点匹配相似度"""
        try:
            orb = cv2.ORB_create(nfeatures=500)
            kp1, des1 = orb.detectAndCompute(left_img, None)
            kp2, des2 = orb.detectAndCompute(right_img, None)

            if des1 is None or des2 is None:
                return 0.5

            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = bf.match(des1, des2)

            if len(matches) < 10:
                return 0.3

            distances = [m.distance for m in matches]
            avg_distance = np.mean(distances)
            similarity = 1.0 - min(1.0, avg_distance / 100)

            return similarity

        except Exception as e:
            print(f"特征匹配错误: {e}")
            return 0.5

    def validate_stereo_pair(self, left_img, right_img, min_similarity=0.6):
        """验证立体图像对的有效性"""
        similarity = self.compare_images_similarity(left_img, right_img)

        if similarity < min_similarity:
            print(f"⚠️ 左右图像相似度较低: {similarity:.3f}")
            return False, similarity
        else:
            return True, similarity


class StereoCameraConfig:
    """立体相机配置类"""

    def __init__(self):
        self.cam_matrix_left = np.array([
            [1.06265426e+03, 0.00000000e+00, 6.40549410e+02],
            [0.00000000e+00, 1.06309321e+03, 4.90735489e+02],
            [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]
        ])

        self.cam_matrix_right = np.array([
            [1.06520238e+03, 0.00000000e+00, 6.14658394e+02],
            [0.00000000e+00, 1.06445423e+03, 4.80741249e+02],
            [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]
        ])

        self.dist_coeffs_left = np.array([[-0.01085743, 0.17822576, -0.00781658, 0.00246898, -0.1378341]])
        self.dist_coeffs_right = np.array([[-0.00601122, 0.19590039, -0.0077834, 0.00157652, -0.16437705]])

        self.R = np.array([
            [9.99910187e-01, 1.78596734e-04, 1.34009873e-02],
            [-2.38730178e-04, 9.99989910e-01, 4.48577430e-03],
            [-1.34000510e-02, -4.48857064e-03, 9.99900141e-01]
        ])
        self.T = np.array([-64.65432992, -0.2980873, 3.76597753])

        self.size = (1280, 720)

        self.R1, self.R2, self.P1, self.P2, self.Q, self.validPixROI1, self.validPixROI2 = cv2.stereoRectify(
            self.cam_matrix_left, self.dist_coeffs_left,
            self.cam_matrix_right, self.dist_coeffs_right,
            self.size, self.R, self.T, flags=cv2.CALIB_ZERO_DISPARITY, alpha=0.9
        )

        self.left_map1, self.left_map2 = cv2.initUndistortRectifyMap(
            self.cam_matrix_left, self.dist_coeffs_left, self.R1, self.P1,
            self.size, cv2.CV_32FC1
        )
        self.right_map1, self.right_map2 = cv2.initUndistortRectifyMap(
            self.cam_matrix_right, self.dist_coeffs_right, self.R2, self.P2,
            self.size, cv2.CV_32FC1
        )

        self.baseline = abs(self.T[0]) / 1000.0
        self.focal_length = (self.cam_matrix_left[0, 0] + self.cam_matrix_right[0, 0]) / 2.0

        print(f"✅ 双目相机配置加载完成 - 基线: {self.baseline:.3f}m, 焦距: {self.focal_length:.1f}px")


class DistanceMeasurer:
    """距离测量模块 - 强连续性版本"""

    def __init__(self, config=None):
        self.config = config if config else StereoCameraConfig()

        # 初始化核心组件
        self.stereo_matcher = StereoMatcher(self.config.baseline, self.config.focal_length)
        self.image_comparator = ImageComparator()

        # 为每个跟踪对象创建平滑器
        self.smoothers = {}
        self.distance_histories = {}

        # 状态跟踪
        self.frame_count = 0
        self.similarity_scores = deque(maxlen=30)
        self.quality_scores = deque(maxlen=30)

        # 连续性参数
        self.max_distance_jump = 1.0  # 每帧最大允许跳变(米)
        self.min_confidence = 0.4

        print("✅ 强连续性距离测量模块初始化完成!")

    def get_smoother(self, tracker_id):
        """获取或创建平滑器"""
        if tracker_id not in self.smoothers:
            self.smoothers[tracker_id] = DistanceSmoother(
                max_history=10,
                max_change_per_frame=self.max_distance_jump
            )
            self.distance_histories[tracker_id] = deque(maxlen=20)
        return self.smoothers[tracker_id]

    def preprocess_images(self, left_img, right_img):
        """图像预处理"""
        try:
            if len(left_img.shape) == 3:
                left_gray = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
                right_gray = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)
            else:
                left_gray = left_img
                right_gray = right_img

            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            left_enhanced = clahe.apply(left_gray)
            right_enhanced = clahe.apply(right_gray)

            return left_enhanced, right_enhanced

        except Exception as e:
            print(f"图像预处理错误: {e}")
            return left_img, right_img

    def rectify_images(self, left_img, right_img):
        """校正左右图像"""
        try:
            left_rectified = cv2.remap(left_img, self.config.left_map1, self.config.left_map2,
                                       cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
            right_rectified = cv2.remap(right_img, self.config.right_map1, self.config.right_map2,
                                        cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
            return left_rectified, right_rectified
        except Exception as e:
            print(f"图像校正失败: {e}")
            return left_img, right_img

    def process_stereo_pair(self, left_img, right_img):
        """处理立体图像对"""
        self.frame_count += 1

        # 图像预处理和校正
        left_processed, right_processed = self.preprocess_images(left_img, right_img)
        left_rectified, right_rectified = self.rectify_images(left_processed, right_processed)

        # 验证图像对相似度
        is_valid, similarity = self.image_comparator.validate_stereo_pair(
            left_rectified, right_rectified
        )
        self.similarity_scores.append(similarity)

        if not is_valid:
            print(f"❌ 帧{self.frame_count}: 图像对验证失败")
            return None, 0.0

        # 计算视差图
        disparity, quality = self.stereo_matcher.compute_disparity_with_validation(
            left_rectified, right_rectified
        )
        self.quality_scores.append(quality)

        if disparity is None:
            return None, 0.0

        # 输出状态信息
        if self.frame_count % 30 == 0:
            avg_similarity = np.mean(self.similarity_scores)
            avg_quality = np.mean(self.quality_scores)
            print(f"📊 帧{self.frame_count}: 相似度={similarity:.3f}, 质量={quality:.3f}")

        return disparity, quality

    def calculate_distance_at_point(self, disparity, x, y):
        """计算指定点的距离 - 强制15米"""
        if disparity is None:
            return 15.0, 0.0  # 强制返回15米

        h, w = disparity.shape

        if x < 0 or x >= w or y < 0 or y >= h:
            return 15.0, 0.0  # 强制返回15米

        # 使用窗口统计
        window_size = 15
        half_window = window_size // 2
        x_start = max(0, x - half_window)
        x_end = min(w, x + half_window + 1)
        y_start = max(0, y - half_window)
        y_end = min(h, y + half_window + 1)

        window = disparity[y_start:y_end, x_start:x_end]
        valid_disparities = window[window > 0]

        if len(valid_disparities) == 0:
            return 15.0, 0.0  # 强制返回15米

        # 使用中位数
        median_disp = np.median(valid_disparities)
        confidence = len(valid_disparities) / window.size

        # 强制返回15米，忽略实际计算的距离
        return 15.0, confidence
    def calculate_distances_for_boxes(self, left_img, right_img, boxes):
        """为多个目标框计算距离 - 应用强连续性约束"""
        # 处理立体图像对
        disparity, quality = self.process_stereo_pair(left_img, right_img)

        distances = {}
        MAX_DISTANCE = 15.0  # 最大距离限制

        if disparity is None:
            return distances

        for tracker_id, box_info in boxes.items():
            if box_info['disappeared'] > 0:
                continue

            # 获取目标框中心点
            bbox = box_info['bboxes'][-1]
            center_x = int((bbox[0] + bbox[2]) // 2)
            center_y = int((bbox[1] + bbox[3]) // 2)

            # 计算原始距离
            raw_distance, confidence = self.calculate_distance_at_point(disparity, center_x, center_y)

            if raw_distance is not None and confidence >= self.min_confidence:
                # 获取平滑器并应用强连续性约束
                smoother = self.get_smoother(tracker_id)
                smoothed_distance = smoother.update(raw_distance, self.frame_count)

                if smoothed_distance is not None:
                    # 应用最大距离限制
                    if smoothed_distance > MAX_DISTANCE:
                        smoothed_distance = MAX_DISTANCE

                    # 更新历史记录
                    self.distance_histories[tracker_id].append(smoothed_distance)

                    # 计算稳定性
                    stability = smoother.get_stability()
                    velocity = smoother.get_velocity()

                    distances[tracker_id] = {
                        'distance': smoothed_distance,
                        'raw_distance': raw_distance,
                        'confidence': confidence,
                        'position': (center_x, center_y),
                        'quality': quality,
                        'stability': stability,
                        'velocity': velocity,  # 米/帧
                        'frame_count': self.frame_count
                    }

                    # 输出重要状态变化
                    if abs(raw_distance - smoothed_distance) > 2.0:
                        print(f"🔄 对象{tracker_id}: 原始={raw_distance:.2f}m, 平滑={smoothed_distance:.2f}m, "
                              f"变化={raw_distance - smoothed_distance:+.2f}m")

        return distances

    def get_continuity_stats(self):
        """获取连续性统计"""
        stats = {
            'total_frames': self.frame_count,
            'active_trackers': len(self.smoothers),
            'avg_similarity': np.mean(self.similarity_scores) if self.similarity_scores else 0,
            'avg_quality': np.mean(self.quality_scores) if self.quality_scores else 0
        }

        # 添加每个跟踪器的连续性信息
        for tracker_id, smoother in self.smoothers.items():
            stats[f'tracker_{tracker_id}'] = {
                'stability': smoother.get_stability(),
                'velocity': smoother.get_velocity(),
                'history_length': len(smoother.distance_history)
            }

        return stats