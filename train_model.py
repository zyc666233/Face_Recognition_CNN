
from dataSet import DataSet
from keras.models import Sequential, load_model, model_from_json
from keras.layers import Dense, Activation, Convolution2D, MaxPooling2D, Flatten, Dropout
import numpy as np


# 建立一个基于CNN的人脸识别模型
class Model(object):
    FILE_PATH = "train_Model\model.h5"  # 模型进行存储和读取的地方
    IMAGE_SIZE = 128  # 模型接受的人脸图片一定得是128*128的

    def __init__(self):
        self.model = None

    # 读取实例化后的DataSet类作为进行训练的数据源
    def read_trainData(self, dataset):
        self.dataset = dataset

    # 建立一个CNN模型，一层卷积、一层池化、一层卷积、一层池化、抹平之后进行全链接、最后进行分类
    def build_model(self):
        self.model = Sequential()
        self.model.add(
            Convolution2D(
                filters=32,
                kernel_size=(5, 5),
                padding='same',
                dim_ordering='th',
                input_shape=self.dataset.X_train.shape[1:]
            )
        )

        self.model.add(Activation('relu'))
        self.model.add(
            MaxPooling2D(
                pool_size=(2, 2),
                strides=(2, 2),
                padding='same'
            )
        )

        self.model.add(Convolution2D(filters=64, kernel_size=(5, 5), padding='same'))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))

        self.model.add(Flatten())
        self.model.add(Dense(512))
        self.model.add(Activation('relu'))

        self.model.add(Dense(self.dataset.num_classes))
        self.model.add(Activation('softmax'))
        self.model.summary()

    # 进行模型训练的函数，具体的optimizer、loss可以进行不同选择
    def train_model(self):
        self.model.compile(
            optimizer='adam',  # 有很多可选的optimizer，例如RMSprop,Adagrad，也可以试试，个人感觉差异不大
            loss='categorical_crossentropy',  # 也可以选用squared_hinge作为loss看看哪个好
            metrics=['accuracy'])

        # epochs、batch_size为可调的参数，epochs为训练多少轮、batch_size为每次训练多少个样本
        self.model.fit(self.dataset.X_train, self.dataset.Y_train, epochs=4, batch_size=60)

    def evaluate_model(self):
        print('\nTesting---------------')
        loss, accuracy = self.model.evaluate(self.dataset.X_test, self.dataset.Y_test)

        print('test loss;', loss)
        print('test accuracy:', accuracy)

    def save(self, file_path=FILE_PATH):
        self.model.save(file_path)
        #self.model.save_weights(file_path)
        print('Model Saved.')

    def load(self, file_path=FILE_PATH):
        #self.model = load_model(file_path)
        self.model.load_weights(file_path, by_name=True)
        json_string = self.model.to_json()
        self.model = model_from_json(json_string)
        print('Model Loaded.')

    # 需要确保输入的img得是灰化之后（channel =1 )且 大小为IMAGE_SIZE的人脸图片
    def predict(self, img):
        img = img.reshape((1, 1, self.IMAGE_SIZE, self.IMAGE_SIZE))
        img = img.astype('float32')
        img = img / 255.0

        result = self.model.predict_proba(img)  # 测算一下该img属于某个label的概率
        max_index = np.argmax(result)  # 找出概率最高的

        return max_index, result[0][max_index]  # 第一个参数为概率最高的label的index,第二个参数为对应概率


if __name__ == '__main__':
    dataset = DataSet('pictures/dataset')
    model = Model()
    model.read_trainData(dataset)
    model.build_model()
    model.train_model()
    model.evaluate_model()
    model.save()
