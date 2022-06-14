#=================================================
#_________ Пример данных MNIST датасета __________
#=================================================

COLS = 5
ROWS = 5

from tensorflow.keras.datasets import mnist
from matplotlib import pyplot as plt
# load dataset
(trainX, trainy), (testX, testy) = mnist.load_data()

# Количественные данные датасета
# X - изображение
# y - разметка
print('Train: X=%s, y=%s' % (trainX.shape, trainy.shape))
print('Test: X=%s, y=%s' % (testX.shape, testy.shape))

for i in range(COLS*ROWS):
	# Отрисовка нескольких изображений на одном
	plt.subplot(COLS, ROWS, i+1)
	plt.imshow(trainX[i], cmap=plt.get_cmap('gray'))
	
plt.show()