from PIL import Image
import os
import matplotlib.pyplot as plt

# 用于格式化图片,批处理图像数据集等
# dst_size: typeof Tuple 目标图片大小,默认为128*128
# suffixList: typeof List 待处理图片文件的后缀列表,可以指明指处理某种或多种
class ImgFormatUtil:
    def __init__(self, dst_size=(128,128), suffixList=['.jpg','.jpeg','.png','.bmp']):
        self.dst_size = dst_size
        self.dst_w, self.dst_h = dst_size
        self.suffixList = suffixList

    # 对单一图片进行resize, 输入参数应为PIL.Image对象或者图像路径
    def imgResize(self, img, path=None, resample=Image.ANTIALIAS):
        if path is not None:
            try:
                img = Image.open(path)
            except IOError:
                return
        w, h = img.size
        # 判断待处理图片比例是否与目标图片比例一致
        if w/h == self.dst_w/self.dst_h:
            # 计算缩放/扩大比例
            if w > self.dst_w:
                n = w / self.dst_w if(w / self.dst_w) >= (h / self.dst_h) else h / self.dst_h
                dst = img.resize((int(w / n), int(h / n)), resample=resample)
            else:
                n = self.dst_w / w if(self.dst_w / w) >= (self.dst_h / h) else self.dst_h / h
                dst = img.resize((int(w * n), int(h * n)), resample=resample)
            return dst
        else:
            # 未保证图像不失真,这里暂时先采用裁剪的方法按照目标图片大小的比例居中裁剪不符合尺寸的源文件
            pass

    # dir_name:源文件目录, resultPath:输出文件根目录
    # 迭代源文件夹, Resize所有img文件后以输出文件根目录为基新建所有对应子文件夹及其子文件
    def imgResize_Batch(self, dir_name, resultPath, resample=Image.ANTIALIAS):
        if not os.path.exists(resultPath):
            os.makedirs(resultPath)

        # 保存顶级目录路径字符串的长度,用以拼接
        region_len = 0

        for root, dirs, files in os.walk(dir_name):
            if(region_len == 0):
                region_len = len(root) + 1
            for filename in files:
                img_name, img_ext = os.path.splitext(filename)
                if img_ext not in self.suffixList:
                    continue
                region = os.path.join(root, filename)
                dstImg = self.imgResize(None, region, resample=resample)
                dst_root_path = os.path.join(resultPath, root[region_len:])
                if not os.path.exists(dst_root_path):
                    os.makedirs(dst_root_path)
                dst_filepath = os.path.join(dst_root_path, img_name + "_"+"%sx%s" % (str(self.dst_w),str(self.dst_h)) + img_ext)
                dstImg.save(dst_filepath, quality = 100)
        


if __name__ == '__main__':
    imUtil = ImgFormatUtil(dst_size=(512,512))
    '''
    img1 = Image.open('./TensorFlow/img/test1.jpeg')
    img2 = imUtil.imgResize(img1)
    img2.save("result.jpg",quality=100)
    plt.figure(figsize=(6,3))
    plt.subplot(1,2,1)
    plt.imshow(img1)
    plt.subplot(1,2,2)
    plt.imshow(img2)
    plt.show()
    '''
    # 使用绝对路径定位当前imgFormatUtil.py文件所在目录,以此目录为根目录进行相对路径查找(测试用工作目录在上一级)
    # 实际使用时只需注意调用该类的py文件所在的工作目录即可
    src_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), './img')
    dst_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), './result/')
    imUtil.imgResize_Batch(src_path, dst_path)