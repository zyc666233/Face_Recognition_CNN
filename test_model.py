import os
from read_image import endwith
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from read_data import read_name_list,read_file
from train_model import Model
import cv2

#读取一张图片进行识别
def test_Pictures(path):
    model= Model()
    model.load()
    for child_dir in os.listdir(path):
        if endwith(child_dir, 'jpg'):  # 找到以jpg结尾的文件路径
            str = os.path.join(path, child_dir)  # 具体的图片文件的路径
            img = cv2.imread(str)
            resized_img = cv2.resize(img, (128, 128),interpolation=cv2.INTER_CUBIC)
            img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)
            picType,prob = model.predict(img)
            if picType != -1:
                name_list = read_name_list('pictures/dataset')
                print (name_list[picType],prob)
            else:
                print (" Don't know this person")

#读取文件夹下子文件夹中所有图片进行识别
def test_onBatch(path):
    model= Model()
    model.load()
    index = 0
    img_list, label_lsit, counter = read_file(path)
    for img in img_list:
        picType,prob = model.predict(img)
        if picType != -1:
            index += 1
            name_list = read_name_list('pictures/dataset')
            print (name_list[picType], prob)
        else:
            print (" Don't know this person")

    return index

if __name__ == '__main__':
    #test_onePicture('pictures/test_photos/zhouyicheng/zhouyicheng_1.jpg')
    #test_onePicture('pictures/test_photos/zhouyicheng/zhouyicheng_2.jpg')
    #test_onePicture('pictures/test_photos/louxingyu/louxingyu_1.jpg')
    #test_onePicture('pictures/test_photos/louxingyu/louxingyu_2.jpg')
    test_Pictures('pictures/take_photos')