from PIL import Image
import numpy as np
import os.path


def generate_noise_gauss(mat_image, mean, std):
    """
    生成高斯噪声图像
    :param mat_image: 原图像矩阵
    :param mean: 均值
    :param std: 标准差
    :return: 加噪图像矩阵
    """
    mat_noise = np.random.normal(mean, std, mat_image.shape)  # 生成高斯噪声矩阵
    mat_image_noise = mat_image + mat_noise  # 原图像加上噪声得到加噪图像
    mat_image_noise = np.clip(mat_image_noise, 0, 255)  # 截取无效值
    mat_image_noise = mat_image_noise.astype('uint8')  # 类型转换
    return mat_image_noise


def generate_noise_salt_pepper(mat_image, amount=.05):
    """
    生成椒盐噪声图像
    :param mat_image: 原图像矩阵
    :param amount: 椒盐噪点占比
    :return: 加噪图像矩阵
    """
    mat_image_noise = np.copy(mat_image)
    num = np.ceil(amount * mat_image.size * .5)
    salt_coords = [np.random.randint(0, i - 1, int(num)) for i in mat_image.shape]
    pepper_coords = [np.random.randint(0, i - 1, int(num)) for i in mat_image.shape]
    mat_image_noise[salt_coords[0], salt_coords[1]] = 255
    mat_image_noise[pepper_coords[0], pepper_coords[1]] = 0
    return mat_image_noise


def save_noise_image_gauss(image_saved, filename_origin):
    name, ext = os.path.splitext(filename_origin)
    print(f"文件名（{name}），扩展名（{ext}）")
    image_saved.save(f"./noise_output/{name}_noise_{mean}_{std}{ext}")
    return


def save_noise_image_pepper_salt(image_saved, filename_origin):
    name, ext = os.path.splitext(filename_origin)
    print(f"文件名（{name}），扩展名（{ext}）")
    image_saved.save(f"./noise_output/{name}_noise_{amount}{ext}")
    return


if __name__ == '__main__':
    path = "./image/lena_gray_512.tif"
    image = Image.open(path).convert("L")
    mat_image = np.array(image)
    # mode="gauss"
    mode="pepper_salt"
    if mode == "gauss":
        mean = 0
        std = 10
        mat_image_noise = generate_noise_gauss(mat_image, mean, std)
        image_noise = Image.fromarray(mat_image_noise)
        save_noise_image_gauss(image_noise, os.path.basename(path))
    elif mode == "pepper_salt":
        amount = 0.05
        mat_image_noise = generate_noise_salt_pepper(mat_image, amount)
        image_noise = Image.fromarray(mat_image_noise)
        save_noise_image_pepper_salt(image_noise, os.path.basename(path))


