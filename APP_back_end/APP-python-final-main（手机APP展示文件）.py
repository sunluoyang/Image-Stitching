# coding=utf-8
from time import time
# import time
# import torch
# from torch import nn, optim
# import torchvision
import os
import cv2
import sys
sys.path.append(r"F:\anaconda\Lib")
sys.path.append(r"F:\anaconda\Lib\site-packages")
sys.path.append(r"F:\anaconda\Lib\site-packages\matplotlib")

import numpy as np
import glob as glob
from  matplotlib import pyplot as plt
import math

# device = torch.device('cuda:5' if torch.cuda.is_available() else 'cpu')

# print(torch.__version__)
print(cv2.__version__)
# print(torchvision.__version__)
# print(device)

import glob

import os
import datetime
# import time
import operator

import pymysql

while True:
    # 文件夹目录
    path = r"C:\Users\ASUS\Desktop\thinkphp5_uploadfile\public\upload"

    # 获取文件夹中所有的文件(名)，以列表形式返货
    lists = os.listdir(path)
    # print("未经处理的文件夹列表：\n %s \n"%lists )

    # 按照key的关键字进行生序排列，lambda入参x作为lists列表的元素，获取文件最后的修改日期，
    # 最后对lists以文件时间从小到大排序
    lists.sort(key=lambda x: os.path.getmtime((path + "\\" + x)))

    # 获取最新文件的绝对路径，列表中最后一个值,文件夹+文件名
    file_new = os.path.join(path, lists[-1])
    # print("时间排序后的的文件夹列表：\n %s \n"%lists )

    # print("最新文件路径:\n%s"%file_new)

    year_n = datetime.datetime.now().year
    year_nem = '%d' % year_n
    year_m = datetime.datetime.now().month
    year_mem = '%d' % year_m
    year_d = datetime.datetime.now().day
    year_dem = '%d' % year_d
    zz = '0'
    count = 0
    if(file_new[-8:]==year_nem+zz+year_mem+year_dem):
        break

print(file_new)
print('新文件夹检索成功')

if(file_new[-8:] == year_nem+zz+year_mem+year_dem):
    while True:

        path_file_number_pre = len(os.listdir(file_new))

        path_file_number_now = len(os.listdir(file_new))

        if(path_file_number_now==0):
            flag = 0
        else:
            flag = 1

        if((flag==1) and (path_file_number_pre==path_file_number_now)):
            count = count + 1
        if(count == 50000):
            break

    print('图像拼接开始')
    # 该参数为阈值，不太清楚具体作用，但是测试中1.10效果比较好
    GOOD_POINTS_LIMITED = 1

    THR = 5


    # 图像拼接
    class Image_mosaic():
        # 初始化
        def __init__(self, fea_extraction, bool_ghost, ghost_reduction, bool_optimize, optimizer, bool_cut):
            '''
            图像拼接类初始化
            :param fea_extraction: 特征点提取算法
            :param bool_ghost: 是否去鬼影
            :param ghost_reduction: 去鬼影算法
            :param bool_optimize: 是否进行对图像进行优化
            :param optimizer: 优化算法
            :param bool_cut: 是否进行图像修剪
            '''
            self.feature_extraction = fea_extraction
            self.optimize_bool = bool_optimize
            self.optimizer = optimizer
            self.ghost_reduction_bool = bool_ghost
            self.ghost_reduction = ghost_reduction
            self.cut_bool = bool_cut
            self.img_final = None
            self.img1 = None
            self.img2 = None
            self.stitch = None

        # 图片大小调整
        def resize_image(self):

            h1, w1, p1 = self.img1.shape
            h2, w2, p2 = self.img2.shape

            h = max(h1, h2)
            w = max(w1, w2)
            self.img1 = cv2.copyMakeBorder(self.img1, int(np.ceil((h - h1) / 2)), int(np.floor((h - h1) / 2)),
                                           int(np.ceil((w - w1) / 2)), int(np.floor((w - w1) / 2)), cv2.BORDER_CONSTANT,
                                           value=(0, 0, 0))
            self.img2 = cv2.copyMakeBorder(self.img2, int(np.ceil((h - h1) / 2)), int(np.floor((h - h1) / 2)),
                                           int(np.ceil((w - w1) / 2)), int(np.floor((w - w1) / 2)), cv2.BORDER_CONSTANT,
                                           value=(0, 0, 0))

        # 图像拼接
        def mosaic(self):
            # print("图像拼接")
            h1, w1, p1 = self.img1.shape  # img1，新图像，w1、h1为宽和高
            h2, w2, p2 = self.img2.shape  # img2，待拼接图像，w2、h2为宽和高

            M1 = np.array([[1., 0., w1], [0., 1., h1]])
            dst1 = cv2.warpAffine(self.img1, M1, (w2 + 2 * w1, h2 + 2 * h1))  # 通过仿射变换将图像
            dst2 = cv2.warpAffine(self.img2, M1, (w2 + 2 * w1, h2 + 2 * h1))  # 置于边长为w2+2*w1的中央

            start_fea = time()
            kp1, des1 = self.stitch.detectAndCompute(dst1, None)  # self.stitch=cv2.xfeatures2d_SURF.
            kp2, des2 = self.stitch.detectAndCompute(dst2, None)  # create(hessianThreshold=100)
            stop_fea = time()
            print("此次特征点提取需要" + str(stop_fea - start_fea) + "秒")

            start_fea = time()
            FLANN_INDEX_KDTREE = 0  # kd树
            index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
            search_params = dict(checks=50)  # or pass empty dictionary
            flann = cv2.FlannBasedMatcher(index_params, search_params)
            matches = flann.knnMatch(des1, des2, k=2)
            stop_fea = time()
            print("此次特征点匹配需要" + str(stop_fea - start_fea) + "秒")

            # 计算新匹配程度
            # bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            # bf = cv2.BFMatcher()
            # matches = bf.match(des1, des2)
            # matches = bf.knnMatch(des1,des2, k=2)
            # print(matches)
            # matches = sorted(matches, key=lambda x: x.distance)
            # 提取goodPoint
            goodPoints = []
            for m, n in matches:
                if m.distance < 0.7 * n.distance:
                    goodPoints.append(m)

            # 画匹配图
            # img3 = cv2.drawMatches(img1, kp1, img2, kp2, goodPoints, flags=2, outImg=None)
            # cv2.imshow(datetime.now().strftime("%Y%m%d_%H%M%S"),img3)

            src_pts = np.float32([kp1[m.queryIdx].pt for m in goodPoints]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in goodPoints]).reshape(-1, 1, 2)

            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

            imageTransform = cv2.warpPerspective(dst1, M, (w2 + 2 * w1, h2 + 2 * h1))

            # print("success")
            self.img_final = np.maximum(dst2, imageTransform)

            print("shape:", self.img_final.shape)

            # plt.figure(1)
            # plt.imshow(self.img_final[:,:,::-1])

            # plt.figure(2)
            # plt.imshow(dst1[:,:,::-1])
            # plt.figure(3)
            # plt.imshow(dst2[:,:,::-1])

        # 图像修剪
        def image_cut(self):
            # print("图像修剪")
            start_cut = time()
            x_start = -1
            x_end = 100000000
            y_start = -1
            y_end = 100000000
            h, w, p = self.img_final.shape
            # 确定y方向图像位置
            for i in range(h):
                # print(str(i)+":",np.sum(self.img_final[i]))
                if np.sum(self.img_final[i]) > THR and y_start < 0:
                    y_start = i
                if y_start > -1 and y_end > 10000000 and np.sum(self.img_final[i]) < THR:
                    y_end = i
            if y_end > 1000000:
                y_end = h
            if y_start < 0:
                y_start = 0
            # 确定x方向图像位置
            for j in range(w):
                if np.sum(self.img_final[:, j]) > 20 and x_start == -1:
                    x_start = j
                if x_start != -1 and x_end > 10000000 and np.sum(self.img_final[:, j]) < 20:
                    x_end = j
            if x_end > 1000000:
                x_end = w
            if x_start < 0:
                x_start = 0
            # 裁剪
            self.img_final = self.img_final[y_start:y_end, x_start:x_end]
            # print("cut")
            stop_cut = time()
            print("此次修剪需要" + str(stop_cut - start_cut) + "秒")

        # 去鬼影(算法和参数后续添加)

        def ghost_way1(self):
            print("ghost_way1")

        def ghost_way2(self):
            print("ghost_way2")

        def ghost_way3(self):
            print("ghost_way3")

        def ghost_Reduction(self):
            # 方法可选(自行填充)
            switch = {"ghost_way1": self.ghost_way1,
                      "ghost_way2": self.ghost_way2,
                      "ghost_way3": self.ghost_way3}
            switch.get(self.ghost_reduction)()

        # 图像融合优化算法(算法和参数自行添加)
        def weight_average(self):
            pass
            # print("weight_average")

        def average(self):
            pass
            # print("average")

        def optimize(self):
            # 方法可选(自行填充)
            switch = {"weight_average": self.weight_average,
                      "average": self.average}
            switch.get(self.optimizer)()

        # 图像拼接
        def creat_orb(self):
            print("creat_orb")
            return cv2.ORB_create()

        def creat_sift(self):
            print("creat_sift")
            return cv2.xfeatures2d.SIFT_create()

        def creat_flann(self):
            print("creat_flann")

        def creat_surf(self):
            print("creat_surf")
            return cv2.xfeatures2d_SURF.create(hessianThreshold=100)

        def mosaic_fusion(self, img1, img2):
            self.img1 = img1
            self.img2 = img2

            # 处理图片大小
            # self.resize_image()

            switch = {"ORB": self.creat_orb,
                      "SIFT": self.creat_sift,
                      "SURF": self.creat_surf,
                      "FLANN": self.creat_flann}
            # switch.get(self.feature_extraction)()

            # 图像拼接
            '''
                class中self.~变量均可不返回（此处写出仅为今后拓展其他算法）
            '''
            self.stitch = switch.get(self.feature_extraction)()
            # 拼接
            self.mosaic()
            # 去鬼影
            if self.ghost_reduction_bool:
                self.ghost_Reduction()
            # 融合
            if self.optimize_bool:
                self.optimize()
            # 图像修剪
            if self.cut_bool:
                self.image_cut()
            # cv2.imshow("mosaic",self.img_final)

            return self.img_final


    file_path = file_new


    class image_stitch():
        # 初始化
        def __init__(self, fea_extraction="SURF", bool_ghost=False, ghost_reduction="ghost_way1", bool_optimize=True,
                     optimizer="average", bool_cut=True):
            self.filename = file_path
            self.image = []
            self.image_gray = []
            self.image_order = []
            self.feature_extraction = fea_extraction
            self.optimize_bool = bool_optimize
            self.optimizer = optimizer
            self.ghost_reduction_bool = bool_ghost
            self.ghost_reduction = ghost_reduction
            self.cut_bool = bool_cut

        def cylindricalWarpImage(self, img1, f=1000, savefig=False):
            # 圆柱形状变换
            im_h, im_w, p = img1.shape
            K = np.array([[f, 0, im_w / 2], [0, f, im_h / 2], [0, 0, 1]])

            # go inverse from cylindrical coord to the image
            # (this way there are no gaps)
            cyl = np.zeros_like(img1)
            cyl_mask = np.zeros_like(img1)
            cyl_h, cyl_w, p = cyl.shape
            x_c = float(cyl_w) / 2.0
            y_c = float(cyl_h) / 2.0
            for x_cyl in np.arange(0, cyl_w):
                for y_cyl in np.arange(0, cyl_h):
                    theta = (x_cyl - x_c) / f
                    h = (y_cyl - y_c) / f

                    X = np.array([math.sin(theta), h, math.cos(theta)])
                    X = np.dot(K, X)
                    x_im = X[0] / X[2]
                    if x_im < 0 or x_im >= im_w:
                        continue

                    y_im = X[1] / X[2]
                    if y_im < 0 or y_im >= im_h:
                        continue

                    cyl[int(y_cyl), int(x_cyl)] = img1[int(y_im), int(x_im)]

            if savefig:
                plt.imshow(cyl[:, :, ::-1])
                plt.savefig("cyl.png", bbox_inches='tight')

            return cyl

        # 下载图片
        def load_img(self):
            # 默认为jpg文件（手机拍摄）
            img_path = glob.glob(file_path + "/*jpg")
            print(img_path)
            # 信息采集程序拍摄图片png文件
            # img_path = glob.glob(file_path + "/*png")
            for i in range(len(img_path)):
                path_s = img_path[i]
                img = cv2.imread(path_s)
                img = self.cylindricalWarpImage(img)
                self.image.append(img)
                img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                self.image_gray.append(img)

        # 图像测序
        def Image_sequencing(self):
            selected_img = []
            for n in range(len(self.image_gray) - 1):
                s_id = -1
                m_id = -1
                best_score = -1

                for i in range(len(self.image_gray)):
                    for j in range(len(self.image_gray)):
                        if i == j:
                            continue
                        if len(selected_img) != 0:
                            if (i != selected_img[-1]) & (i != selected_img[1]) & (i != selected_img[-2]):
                                continue
                            if j in selected_img:
                                continue

                        I = np.fft.fft2(self.image_gray[i])
                        J = np.fft.fft2(self.image_gray[j])
                        C = np.fft.ifft2((I * np.conj(J)) / np.sqrt(I * np.conj(I) * J * np.conj(J)))
                        C = np.abs(C)
                        score = np.max(C)
                        # print(i, " ", j, " ", score)
                        if score > best_score:
                            best_score = score
                            s_id = i
                            m_id = j

                selected_img.append(s_id)
                selected_img.append(m_id)

            self.image_order = []
            [self.image_order.append(i) for i in selected_img if not i in self.image_order]
            # print(self.image_order)

        # 运行
        def run(self):
            start = time()
            self.load_img()
            stop = time()
            print("图片读取并修饰时间" + str(stop - start) + "秒")
            start = time()
            self.Image_sequencing()
            stop = time()
            print(self.image_order)
            print("测序时间" + str(stop - start) + "秒")

            # x = input("请输入1<=x<len()-1以查看第x次拼接结果：")
            x = 10

            final_image = None
            # 构建拼接算法
            mosaic = Image_mosaic(fea_extraction=self.feature_extraction, bool_ghost=self.ghost_reduction_bool,
                                  ghost_reduction=self.ghost_reduction,
                                  bool_optimize=self.optimize_bool, optimizer=self.optimizer, bool_cut=self.cut_bool)

            for i in range(1, len(self.image_order)):
                # for i in range(1,20):
                if i == 1:
                    start = time()
                    final_image = mosaic.mosaic_fusion(self.image[self.image_order[1]], self.image[self.image_order[0]])
                    stop = time()
                    plt.figure("最终拼接结果")
                    plt.imshow(final_image[:, :, ::-1])
                    print("第" + str(i) + "次拼接需要" + str(stop - start) + "秒")
                else:
                    start = time()
                    final_image = mosaic.mosaic_fusion(self.image[self.image_order[i]], final_image)
                    stop = time()
                    plt.figure("最终拼接结果")
                    plt.imshow(final_image[:, :, ::-1])
                    print("第" + str(i) + "次拼接需要" + str(stop - start) + "秒")
                if i == int(x):
                    plt.figure("第" + str(x) + "次拼接图")
                    plt.imshow(final_image[:, :, ::-1])

            '''
            final_image = mosaic.mosaic_fusion(self.image[2], self.image[1])
            cv2.imshow("mind",final_image)
            final_image = mosaic.mosaic_fusion(final_image, self.image[3])
            final_image = mosaic.mosaic_fusion(final_image, self.image[0])
            '''

            cv2.imwrite("final_im.png", final_image)
            plt.imshow(final_image[:, :, ::-1])


    if __name__ == "__main__":
        start_all = time()
        # 通过传入参数改变拼接方法（空为默认参数）
        image_stitch().run()
        stop_all = time()
        print("程序运行需要" + str(stop_all - start_all) + "秒")
    import time, random

    import sys
    import pymysql
    from PIL import Image
    import os
    import base64

    # 读取图片文件
    fp = open(r"F:\pycharm\工程\APP工程\APP-try1\final_im.png", 'rb')
    img = fp.read()
    fp.close()

    newName = time.strftime('%Y%m%d%H%M%S') + '_%d' % random.randint(100, 1000)  # 定义文件名，年月日时分秒随机数
    newFile = r"C:\Users\ASUS\Desktop\thinkphp5_uploadfile\public\upload/finish/" + newName + ".jpg"
    newUrl = "http://192.168.32.71:8081/upload/finish/" + newName + ".jpg"
    with open(newFile, 'wb') as destination:
        destination.write(img)

    database = pymysql.connect(host="127.0.0.1", user="root", passwd="root", db="img", charset='utf8')
    cursor = database.cursor()
    query = """INSERT INTO newimg(cover) VALUES ("{}")""".format(newUrl)
    cursor.execute(query)

    database.commit()
    # 关闭游标
    cursor.close()
    # 关闭数据库连接
    database.close()
    print("============")
    print("图片已经上传数据库")