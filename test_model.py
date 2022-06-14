#=================================================
#______Загрузкаи преобразование изображения_______
#_____________для подачи в модель_________________
#=================================================

from numpy import argmax
from keras.utils import load_img
from keras.utils import img_to_array
from keras.models import load_model

def load_image(filename):
	# Загружаем изображение
	img = load_img(filename, grayscale=True, target_size=(28, 28))
	# Преобразуем в массив
	img = img_to_array(img)
	# Изменяеми размер и канал
	img = img.reshape(1, 28, 28, 1)
	# Нормализуем
	img = img.astype('float32')
	img = img / 255.0
	return img

def predict(model_name, img):
	img = load_image(img)
	model = load_model(model_name)
	predict_value = model.predict(img)
	#Ищем индекс с максимальным откликом
	digit = argmax(predict_value)
	return digit
