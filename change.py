import cv2
import numpy as np
import skimage.restoration 

# 讀取影像
img_A = cv2.imread('messageImage_1721811902467.jpg')
img_B = cv2.imread('image_crater0_0.jpg')

# 影像放大
img_A_resized = cv2.resize(img_A, (3000, 1220), interpolation=cv2.INTER_CUBIC)

# 轉換為HSV色彩空間
hsv_A = cv2.cvtColor(img_A_resized, cv2.COLOR_BGR2HSV)
hsv_B = cv2.cvtColor(img_B, cv2.COLOR_BGR2HSV)

# 提取V通道
V_A = hsv_A[:, :, 2]
V_B = hsv_B[:, :, 2]

def calculate_local_ratio(img_A, img_B, block_size=9):
    """
    1.
    計算每個 3x3 方格的中心像素與影像 B 對應位置的比例，
    並將該比例應用於整個方格。

    Args:
        img_A: 影像 A
        img_B: 影像 B
        block_size: 方格大小 (必須為奇數)

    Returns:
        ratio: 形狀與 img_A 相同的比例圖
    """

    h, w = img_A.shape[:2]
    ratio = np.ones_like(img_A, dtype=np.float32)

    # 確保 block_size 是奇數
    assert block_size % 2 == 1, "block_size must be odd"

    half_size = block_size // 2
    for i in range(half_size, h-half_size):
        for j in range(half_size, w-half_size):
            # 提取 3x3 方格
            block_A = img_A[i-half_size:i+half_size+1, j-half_size:j+half_size+1]
            block_B = img_B[i-half_size:i+half_size+1, j-half_size:j+half_size+1]

            # 計算中心像素的比例
            center_ratio = block_A[half_size, half_size] / block_B[half_size, half_size]

            # 將比例應用於整個方格
            ratio[i-half_size:i+half_size+1, j-half_size:j+half_size+1] = center_ratio

    return ratio

def calculate_each_ratio(V_A, V_B):
    # 2. 計算比例 (這裡採用簡單的逐像素相除)
    ratio = V_A / V_B
    ratio[np.isnan(ratio)] = 1  # 處理除以0的情況

    return ratio

def histogram_matching(source, template):
    """
    Histogram matching to transfer the luminance distribution of the source image to the template image.

    Args:
        source: The source image.
        template: The template image.

    Returns:
        The matched image.
    """

    # Convert images to grayscale
    hsv_A = cv2.cvtColor(img_A, cv2.COLOR_BGR2HSV)
    hsv_B = cv2.cvtColor(img_B, cv2.COLOR_BGR2HSV)

    # Calculate the histograms
    source_hist = cv2.calcHist([hsv_A], [0], None, [256], [0, 256])
    template_hist = cv2.calcHist([hsv_B], [0], None, [256], [0, 256])

    # Calculate the cumulative distribution functions (CDFs)
    source_cdf = source_hist.cumsum()
    source_cdf_normalized = source_cdf * 255 / source_cdf.max()
    template_cdf = template_hist.cumsum()

    # Create a lookup table
    lookup_table = np.zeros(256, dtype=np.uint8)
    for i in range(256):
        # Find the closest value in the template CDF
        j = np.argmin(np.abs(template_cdf - i * template_cdf.max() / 255))
        lookup_table[i] = j

    # Map the source image to the template using the lookup table
    matched_image = cv2.LUT(img_A, lookup_table)

    return matched_image

def histogram_matching(source, template):
    # Convert images to HSV color space
    source_hsv = cv2.cvtColor(source, cv2.COLOR_BGR2HSV)
    template_hsv = cv2.cvtColor(template, cv2.COLOR_BGR2HSV)

    # Extract V channels
    source_v = source_hsv[:, :, 2]
    template_v = template_hsv[:, :, 2]

    # Calculate mean values
    mean_source = np.mean(source_v)
    mean_template = np.mean(template_v)

    # Calculate ratio
    ratio = mean_source / mean_template

    # Adjust template V channel using the ratio
    template_v_adjusted = template_v * ratio

    # Clip values to ensure they are within the valid range (0-255)
    template_v_adjusted = np.clip(template_v_adjusted, 0, 255).astype(np.uint8)

    # Merge the adjusted V channel with the original H and S channels
    template_hsv[:, :, 2] = template_v_adjusted
    result = cv2.cvtColor(template_hsv, cv2.COLOR_HSV2BGR)

    return result

# 計算比例
ratio = calculate_local_ratio(V_A, V_B)

# 調整影像B的V通道
hsv_B[:, :, 2] = V_B * ratio

# 轉回BGR色彩空間
img_B_adjusted = cv2.cvtColor(hsv_B, cv2.COLOR_HSV2BGR)

#3.使用直方圖
result = histogram_matching(img_A, img_B)

# 儲存結果
cv2.imwrite('result3.jpg', img_B_adjusted)