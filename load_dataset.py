#=================================================
#____ Загрузка  настройка параметров датасета ____
#=================================================

from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

# Загрузка тренировочного и тестового датасета
def load_dataset():
	# Загрузка датасета
	(trainX, trainY), (testX, testY) = mnist.load_data()
	# Изменяем датасет, оставляя только один каанал из-я
	trainX = trainX.reshape((trainX.shape[0], 28, 28, 1))
	testX = testX.reshape((testX.shape[0], 28, 28, 1))
	# Преобразование элемента класса (число) в 
    # одномерный двоичный вектор, длинной кол-ва
    # определяемых классов (10)
	trainY = to_categorical(trainY)
	testY = to_categorical(testY)
	return trainX, trainY, testX, testY
