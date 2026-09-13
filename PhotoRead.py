import cv2
import numpy as np
import os


def extract_red_box_advanced(image_path, min_box_area=1000):
    image = cv2.imread(image_path)
    if image is None:
        return None

    # 方法1: HSV颜色空间检测
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 100, 100])
    upper_red2 = np.array([180, 255, 255])

    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = mask1 + mask2

    # 方法2: BGR颜色空间检测
    b, g, r = cv2.split(image)
    red_condition = (r > 150) & (g < 100) & (b < 100)
    bgr_red_mask = np.uint8(red_condition) * 255

    # 合并两种方法的掩码
    combined_mask = cv2.bitwise_or(red_mask, bgr_red_mask)

    # 形态学操作
    kernel = np.ones((5, 5), np.uint8)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)

    # 查找轮廓
    contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        print("未检测到红色框")
        return None

    # 筛选符合条件的轮廓
    valid_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_box_area:
            continue

        # 获取边界矩形
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / h

        # 筛选可能是框的轮廓（长宽比在合理范围内）
        if 0.2 < aspect_ratio < 5.0:
            valid_contours.append(contour)

    if not valid_contours:
        print("未找到合适的红色框")
        return None

    # 选择最大的有效轮廓
    largest_contour = max(valid_contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)

    # 提取区域（向内收缩）
    margin = min(w, h) // 10  # 动态边距
    roi_x = x + margin
    roi_y = y + margin
    roi_w = w - 2 * margin
    roi_h = h - 2 * margin

    # 边界检查
    roi_x = max(0, roi_x)
    roi_y = max(0, roi_y)
    roi_w = min(roi_w, image.shape[1] - roi_x)
    roi_h = min(roi_h, image.shape[0] - roi_y)

    if roi_w > 0 and roi_h > 0:
        extracted_region = image[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w]
        return extracted_region
    else:
        print("提取区域无效")
        return None


# 将图片尺寸设置成一样
def smart_resize_with_padding(image, target_size=(640, 480), padding_color=(0, 0, 0)):
    """
    智能调整图片尺寸，保持宽高比并用指定颜色填充

    Args:
        image: 输入图片
        target_size: 目标尺寸 (width, height)
        padding_color: 填充颜色 (B, G, R)

    Returns:
        调整后的图片
    """
    target_w, target_h = target_size
    if image is not None:
        h, w = image.shape[:2]

        # 计算缩放比例
        scale = min(target_w / w, target_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)

        # 缩放图片
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # 创建填充背景
        if len(image.shape) == 3:  # 彩色图片
            padded = np.full((target_h, target_w, 3), padding_color, dtype=np.uint8)
        else:  # 灰度图片
            padded = np.full((target_h, target_w), padding_color[0], dtype=np.uint8)

        # 计算放置位置（居中）
        x_offset = (target_w - new_w) // 2
        y_offset = (target_h - new_h) // 2

        # 将图片放到背景上
        if len(image.shape) == 3:
            padded[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized
        else:
            padded[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized
        return padded, (x_offset, y_offset, new_w, new_h)
    else:
        return None,()




def read_imgs(base_path, output_path="extracted_images"):
    # 确保输出目录存在
    os.makedirs(output_path, exist_ok=True)
    print(f"📁 输出目录: {output_path}")
    
    # 处理所有图片
    for filename in os.listdir(base_path):
        src_path = os.path.join(base_path, filename)
        print(f"🔍 处理图片: {filename}")
        
        # 使用截取标注图像
        result = extract_red_box_advanced(src_path)
        
        # 将所有图像设置成相同尺寸
        resized, bbox = smart_resize_with_padding(result, (640, 480), (0, 0, 0))
        
        if resized is not None:
            # 生成输出文件名
            name, ext = os.path.splitext(filename)
            output_filename = f"{name}_extracted{ext}"
            output_filepath = os.path.join(output_path, output_filename)
            
            # 保存提取的区域
            cv2.imwrite(output_filepath, resized)
            print(f"✅ 已保存到: {output_filename}")
            
            # 显示结果（可选）
            # cv2.imshow("Extracted Region", resized)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
        else:
            print(f"❌ 无法提取 {filename} 中的红色框")
    
    print(f"\n✅ 处理完成，共处理 {len(os.listdir(base_path))} 张图片")

# 运行函数
read_imgs(r"custom_images")
