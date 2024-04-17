import os.path
from PIL import Image
import numpy as np
import cv2


def build_matrix_dist_square(kernel_size):
    """
    构建距离平方矩阵
    :param kernel_size: 核大小
    :return: 距离平方矩阵
    """
    kernel_radius = kernel_size // 2
    rows, cols = np.indices((kernel_size, kernel_size))
    mat_dist = (rows - kernel_radius) ** 2 + (cols - kernel_radius) ** 2
    return mat_dist


def build_matrix_exponent_space(kernel_size, sigma_s):
    """
    构建空间权重指数矩阵
    :param kernel_size: 核大小
    :param sigma_s: 空间域σ
    :return: 空间权重指数矩阵
    """
    mat_dist_square = build_matrix_dist_square(kernel_size)  # 生成距离平方矩阵
    print(f"距离平方矩阵{mat_dist_square}")
    mat_exponent_space = mat_dist_square / ((sigma_s ** 2) * -2)  # 空间权重指数
    return mat_exponent_space


def handle_neighbor(mat_image, mat_image_pad, kernel_size, sigma_c, x, y):
    """
    处理邻域
    :param mat_image: 图像矩阵
    :param mat_image_pad: 补0图像矩阵
    :param kernel_size: 卷积核大小
    :param sigma_c: 值域σ
    :param x: x
    :param y: y
    :return: 值域权重指数矩阵，邻域灰度矩阵
    """
    # 邻域灰度
    mat_color = mat_image_pad[x:x + kernel_size, y:y + kernel_size]

    # 值域权重指数
    mat_exponent_color = np.full((kernel_size, kernel_size), mat_image[x, y]).astype(float)
    mat_exponent_color = abs(
        mat_exponent_color.astype(int) - mat_color.astype(int))
    mat_exponent_color = np.square(mat_exponent_color)
    mat_exponent_color = mat_exponent_color / ((sigma_c ** 2) * -2)

    return mat_exponent_color, mat_color


def bilateral_filter(mat_image, kernel_size, sigma):
    """
    双边滤波器
    :param mat_image: 原图像矩阵
    :param kernel_size: 核大小
    :param sigma: 滤波参数σ
    :return: 双边滤波图像矩阵
    """
    print(f"Shape为{mat_image.shape}")
    kernel_radius = kernel_size // 2
    mat_image_pad = np.pad(mat_image, ((kernel_radius, kernel_radius), (kernel_radius, kernel_radius)))  # 补0图像矩阵
    mat_exponent_space = build_matrix_exponent_space(kernel_size, sigma[0])  # 空间权重指数
    mat_image_filtered = np.zeros_like(mat_image)  # 过滤图像矩阵
    # 遍历每个像素进行滤波操作
    for x in range(mat_image.shape[0]):
        for y in range(mat_image.shape[1]):
            print(f"像素{x},{y}，值{mat_image[x, y]}")
            mat_exponent_color, mat_color = handle_neighbor(mat_image, mat_image_pad, kernel_size, sigma[1], x,
                                                            y)  # 处理邻域
            mat_weight = np.exp(mat_exponent_space + mat_exponent_color)  # 计算权重矩阵
            mat_image_filtered[x][y] = int(round(np.sum(np.multiply(mat_weight, mat_color)) / np.sum(mat_weight)))
    return mat_image_filtered


def save_filtered_image(image_filtered, filename_origin, kernel_size, sigma):
    name, ext = os.path.splitext(filename_origin)
    path_save = f"./output/{name}_filbil_{kernel_size}_{sigma[0]}_{sigma[1]}{ext}"
    print(f"保存 {path_save}")
    image_filtered.save(path_save)
    return


def filter_and_save(mode, path, kernel_size, sigma):
    if mode == "custom":
        image = Image.open(path).convert("L")  # 加载图像并转换为灰度图像
        mat_image = np.array(image)  # 将图像转换为矩阵
        mat_image_filtered = bilateral_filter(mat_image, kernel_size, sigma)  # 双边滤波

        image_filtered = Image.fromarray(mat_image_filtered)
        image_filtered.show()  # 展示双边滤波图像
        save_filtered_image(image_filtered, os.path.basename(path), kernel_size, sigma)  # 保存双边滤波图像
    elif mode == "opencv":
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        # image_filtered = cv2.medianBlur(image, 3)
        image_filtered = cv2.bilateralFilter(image, d=kernel_size, sigmaColor=sigma[1], sigmaSpace=sigma[0])
        name, ext = os.path.splitext(os.path.basename(path))
        path_save = f"./opencv_output/{name}_filbil_{kernel_size}_{sigma[0]}_{sigma[1]}{ext}"
        cv2.imwrite(path_save, image_filtered)
