
import os
import cv2
import numpy as np
from read_image import endwith

#输入一个文件路径，对其下的每个文件夹下的图片读取，并对每个文件夹给一个不同的Label
#返回一个img的list,返回一个对应label的list,返回一下有几个文件夹（有几种label)

def read_file(path):
    img_list = []
    label_list = []
    dir_counter = 0 #记录文件夹的数量
    IMG_SIZE = 128

    #对路径下的所有子文件夹中的所有jpg文件进行读取并存入到一个list中
    for child_dir in os.listdir(path):
         child_path = os.path.join(path, child_dir) #得到path文件夹啊中的子文件路径

         for dir_image in  os.listdir(child_path):
             if endwith(dir_image,'jpg'):#找到以jpg结尾的文件路径
                str = os.path.join(child_path, dir_image)#具体的图片文件的路径
                #print(str)
                img = cv2.imread(str)
                #print(type(img))
                resized_img = cv2.resize(img, (IMG_SIZE, IMG_SIZE),interpolation=cv2.INTER_CUBIC)
                recolored_img = cv2.cvtColor(resized_img,cv2.COLOR_BGR2GRAY)
                img_list.append(recolored_img)
                label_list.append(dir_counter)

         dir_counter += 1

    # 返回的img_list转成了 np.array的格式
    img_list = np.array(img_list)

    return img_list,label_list,dir_counter

#读取训练数据集的文件夹，把他们的名字返回给一个list
def read_name_list(path):
    name_list = []
    for child_dir in os.listdir(path):
        name_list.append(child_dir)
    return name_list

if __name__ == '__main__':
    img_list,label_list,counter = read_file("pictures\dataset")
    print(counter)
    print(img_list)
    name_list = read_name_list('pictures\dataset')
    print(name_list)

