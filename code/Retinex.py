import cv2
import numpy as np
import math
import argparse
import os
from tqdm import tqdm
import time


class GaussianBlurConv():
    '''
    高斯滤波
    依据图像金字塔和高斯可分离滤波器思路加速
    '''
    def FilterGaussian(self, img, sigma):
        '''
        高斯分离卷积，按照x轴y轴拆分运算，再合并，加速运算
        '''
        # reject unreasonable demands
        if sigma > 300:
            sigma = 300
        # 获取滤波器尺寸且强制为奇数
        kernel_size = round(sigma * 3 * 2 +1) | 1   # 当图像类型为CV_8U的时候能量集中区域为3 * sigma,
        # 创建内核
        kernel = cv2.getGaussianKernel(ksize=kernel_size, sigma=sigma, ktype=cv2.CV_32F)
        # 初始化图像
        temp = np.zeros_like(img)
        # x轴滤波
        for j in range(temp.shape[0]):
            for i in range(temp.shape[1]):
                # 内层循环展开
                v1 = v2 = v3 = 0
                for k in range(kernel_size):
                    source = math.floor(i+ kernel_size/2 -k)        # 把第i个坐标和kernel的中心对齐 -k是从右往左遍历kernel对应的图像，得到与kernel的第k个元素相乘的图像坐标
                    if source < 0:
                        source = source * -1            # 如果图像超出左边缘，就反向，对称填充
                    if source > img.shape[1]:
                        source = math.floor(2 * (img.shape[1] - 1) - source)   # 图像如果超出右边缘，就用左边从头数着补
                    v1 += kernel[k] * img[j, source, 0]
                    if temp.shape[2] == 1: continue
                    v2 += kernel[k] * img[j, source, 1]
                    v3 += kernel[k] * img[j, source, 2]
                temp[j, i, 0] = v1
                if temp.shape[2] == 1: continue
                temp[j, i, 1] = v2
                temp[j, i, 2] = v3
        # 分离滤波，先在原图用x轴的滤波器滤波，得到temp图，再用y轴滤波在temp图上滤波，结果一致
        # y轴滤波
        for i in range(img.shape[1]):         # height
            for j in range(img.shape[0]):
                v1 = v2 = v3 = 0
                for k in range(kernel_size):
                    source = math.floor(j + kernel_size/2 - k)
                    if source < 0:
                        source = source * -1
                    if source > temp.shape[0]:
                        source = math.floor(2 * (img.shape[0] - 1) - source)   # 上下对称
                    v1 += kernel[k] * temp[source, i, 0]
                    if temp.shape[2] == 1: continue
                    v2 += kernel[k] * temp[source, i, 1]
                    v3 += kernel[k] * temp[source, i, 2]
                img[j, i, 0] = v1
                if img.shape[2] == 1: continue
                img[j, i, 1] = v2
                img[j, i, 2] = v3
        return img

    def FastFilter(self, img, sigma):
        '''
        快速滤波，按照图像金字塔，逐级降低图像分辨率，对应降低高斯核的sigma，
        当sigma转换成高斯核size小于10，再进行滤波，后逐级resize
        递归思路
        '''
        # reject unreasonable demands
        if sigma > 300:
            sigma = 300
        # 获取滤波尺寸，且强制为奇数
        kernel_size = round(sigma * 3 * 2 + 1) | 1  # 当图像类型为CV_8U的时候能量集中区域为3 * sigma,
        # 如果s*sigma小于一个像素，则直接退出
        if kernel_size < 3:
            return
        # 处理方式(1) 滤波  (2) 高斯光滑处理  (3) 递归处理滤波器大小
        if kernel_size < 10:
            # img = self.FilterGaussian(img, sigma)
            img = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)   # 官方函数
            return img
        else:
            # 若降采样到最小，直接退出
            if img.shape[1] < 2 or img.shape[0] < 2:
                return img
            sub_img = np.zeros_like(img)        # 初始化降采样图像
            sub_img = cv2.pyrDown(img, sub_img)           # 使用gaussian滤波对输入图像向下采样，缩放二分之一，仅支持CV_GAUSSIAN_5x5
            sub_img = self.FastFilter(sub_img, sigma/2.0)
            img = cv2.resize(sub_img, (img.shape[1], img.shape[0]))              # resize到原图大小
            return img

    def __call__(self, x, sigma):
        x = self.FastFilter(img, sigma)
        return x


class Retinex(object):
    """
    SSR: baseline
    MSR: keep the high fidelity and the dynamic range as well as compressing img
    MSRCR_GIMP:
      Adapt the dynamics of the colors according to the statistics of the first and second order.
      The use of the variance makes it possible to control the degree of saturation of the colors.
    """
    def __init__(self, model='MSR', sigma=[30, 150, 300], restore_factor=2.0, color_gain=10.0, gain=270.0, offset=128.0):
        self.model_list = ['SSR','MSR']
        if model in self.model_list:
            self.model = model
        else:
            raise ValueError
        self.sigma = sigma        # 高斯核的方差
        # 颜色恢复
        self.restore_factor = restore_factor     # 控制颜色修复的非线性
        self.color_gain = color_gain             # 控制颜色修复增益
        # 图像恢复
        self.gain = gain           # 图像像素值改变范围的增益
        self.offset = offset       # 图像像素值改变范围的偏移量
        self.gaussian_conv = GaussianBlurConv()   # 实例化高斯算子

    def _SSR(self, img, sigma):
        filter_img = self.gaussian_conv(img, sigma)    # [h,w,c]
        retinex = np.log10(img) - np.log10(filter_img)
        return retinex

    def _MSR(self, img, simga):
        retinex = np.zeros_like(img)
        for sig in simga:
            retinex += self._SSR(img, sig)
        retinex = retinex / float(len(self.sigma))
        return retinex

    def _colorRestoration(self, img, retinex):
        img_sum = np.sum(img, axis=2, keepdims=True)  # 在通道层面求和
        # 颜色恢复
        # 权重矩阵归一化 并求对数，得到颜色增益
        color_restoration = np.log10((img * self.restore_factor / img_sum) * 1.0 + 1.0)
        # 将Retinex做差后的图像，按照权重和颜色增益重新组合
        img_merge = retinex * color_restoration * self.color_gain
        # 恢复图像
        img_restore = img_merge * self.gain + self.offset
        return img_restore

    def _simplestColorBalance(self, img, low_clip, high_clip):
        total = img.shape[0] * img.shape[1]
        for i in range(img.shape[2]):
            unique, counts = np.unique(img[:, :, i], return_counts=True)  # 返回新列表元素在旧列表中的位置，并以列表形式储存在s中
            current = 0
            for u, c in zip(unique, counts):
                if float(current) / total < low_clip:
                    low_val = u
                if float(current) / total < high_clip:
                    high_val = u
                current += c

            img[:, :, i] = np.maximum(np.minimum(img[:, :, i], high_val), low_val)

        return img

    def _MSRCR_GIMP(self, img):

        # self.img = results['img']
        self.img = np.float32(img) + 1.0
        if self.model == 'SSR':
            self.retinex = self._SSR(self.img, self.sigma)
        elif self.model == 'MSR':
            self.retinex = self._MSR(self.img, self.sigma)
        # 颜色恢复 图像恢复
        self.img_restore = self._colorRestoration(self.img, self.retinex)

        return self.img_restore

    def __call__(self, img):
        return self._MSRCR_GIMP(img)

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += '{},sigma={},dynamic={}'.format(self.model, self.sigma, self.Dynamic)
        return repr_str

def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('--configs', default='configs/dcn/10_1.py', help='train config file path')
    args = parser.parse_args()
    return args

if __name__ == '__main__':

    img_root =r'F:\study\0_Project\sea_detection\retinex\retinex\test\000239.jpg'
    # img_root = r'F:\study\0_Project\sea_detection\MSRCR-Restoration-master\image\1.png'
    img = cv2.imread(img_root)
    retinex = Retinex()
    img = retinex(img)
    save_root = r'F:\study\0_Project\sea_detection\retinex\retinex\test\000239_test.jpg'
    cv2.imwrite(save_root, img)
    # 读数据
    # img_root = '../data/seacoco/train/'
    # img_list = os.listdir(img_root)
    # img_save = img_root.replace('train/', 'train_lable')
    # if not os.path.exists(img_save):
    #     os.makedirs(img_save)
    # retinex = Retinex()
    # start = time.time()
    #
    # for img_name in tqdm(img_list):
    #     read_root = img_root + img_name
    #     save_root = img_save + '/' + img_name
    #     img = cv2.imread(read_root)
    #     img = retinex(img)
    #     cv2.imwrite(save_root, img)
    # end = time.time()
    # print('total_time: {}'.format(end - start))
