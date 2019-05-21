import cv2
import math
import numpy as np
import os
# 计算图像的暗通道

def getdark(im,sz):
    im = np.array(im)
    b,g,r = np.split(im,1,axis=2)

def DarkChannel(im, sz):
    b, g, r = cv2.split(im)         #取 R,G,B通道
    dc = cv2.min(cv2.min(r, g), b)          #取三通道的最小值作为暗通道
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (sz, sz))
    dark = cv2.erode(dc, kernel)    #腐蚀暗通道
    # dark = dc
    return dark

def AtmLight(im, dark):
    [h, w] = im.shape[:2]       #dark为RGB中最小的值
    imsz = h * w                #获取宽高
    numpx = int(max(math.floor(imsz / 1000), 1))    # 像素规模/1000？
    darkvec = dark.reshape(imsz, 1)     #暗通道    flatten
    imvec = im.reshape(imsz, 3)         #RGB
    indices = darkvec.argsort(axis=0)     #获取数组从小到大的值索引
    indices = indices[imsz - numpx::]   #获取第imsz-numpx之后的所有值

    atmsum = np.zeros([1, 3])
    for ind in range(1, numpx):
        atmsum = atmsum + imvec[indices[ind]]
    A = atmsum / numpx
    return A

def TransmissionEstimate(im, A, sz):
    omega = 0.95
    im3 = np.empty(im.shape, im.dtype)

    for ind in range(0, 3):
        im3[:, :, ind] = im[:, :, ind] / A[0, ind]

    transmission = 1 - omega * DarkChannel(im3, sz)
    return transmission

def Guidedfilter(im, p, r, eps):
    mean_I = cv2.boxFilter(im, cv2.CV_64F, (r, r))
    mean_p = cv2.boxFilter(p, cv2.CV_64F, (r, r))
    mean_Ip = cv2.boxFilter(im * p, cv2.CV_64F, (r, r))
    cov_Ip = mean_Ip - mean_I * mean_p

    mean_II = cv2.boxFilter(im * im, cv2.CV_64F, (r, r))
    var_I = mean_II - mean_I * mean_I

    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I

    mean_a = cv2.boxFilter(a, cv2.CV_64F, (r, r))
    mean_b = cv2.boxFilter(b, cv2.CV_64F, (r, r))

    q = mean_a * im + mean_b
    return q

def TransmissionRefine(im, et):
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    gray = np.float64(gray) / 255
    r = 60
    eps = 0.0001
    t = Guidedfilter(gray, et, r, eps)
    return t

def Recover(im, t, A, tx=0.1):
    res = np.empty(im.shape, im.dtype)
    t = cv2.max(t, tx)

    for ind in range(0, 3):
        res[:, :, ind] = (im[:, :, ind] - A[0, ind]) / t + A[0, ind]

    return res

if __name__ == '__main__':
    fn_path = "/home/yel/data/Aerialgoaf/detection/test"
    dst_path="/home/yel/data/Aerialgoaf/detection/test/haze"
    for i in os.listdir(fn_path):
        if 'xml' in i:
            continue
        if 'haze' in i:
            continue
        fn=os.path.join(fn_path,i)
        src = cv2.imread(fn,cv2.IMREAD_COLOR)
        I = src.astype('float64') / 255
        dark = DarkChannel(I, 15)
        A = AtmLight(I, dark)
        te = TransmissionEstimate(I, A, 15)
        t = TransmissionRefine(src, te)
        J = Recover(I, t, A, 0.1)
        # dark_show=dark*255
        # dark_show=dark_show.astype('uint8')
        # cv2.imshow("darkChannel",dark_show)
        cv2.imwrite(os.path.join(dst_path,i.split('.')[0]+'.png'), J * 255)
        cv2.waitKey()
