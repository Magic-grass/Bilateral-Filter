import bilateral_filter
import threading


def print_hi():
    print(f'Hi! Hi! Hi! 鸡汤来咯~')
    return


if __name__ == '__main__':
    # path = "image/lena_gray_512.tif"
    # path = "./noise_output/lena_gray_512_noise_0_10.tif"
    path = "./noise_output/lena_gray_512_noise_0.05.tif"
    kernel_size = 11  # 卷积核大小
    # 标准差
    sigmas = [(5, 30)]
    # sigmas = [(8, 10), (8, 30), (8, 50), (8, 100), (8, 200), (8, 255)]
    mode = "custom"
    # mode = "opencv"
    for sigma in sigmas:
        threading.Thread(target=bilateral_filter.filter_and_save, args=(mode, path, kernel_size, sigma)).start()
