#=================================================
#_______ Нормализация пикселей изображения _______
#_________________ от 0 до 1 _____________________
#=================================================

# Нормализация изображений
def normalize_pixels(train, test):
	# int to float
	train_norm = train.astype('float32')
	test_norm = test.astype('float32')
	# Нормализация
	train_norm = train_norm / 255.0
	test_norm = test_norm / 255.0
	# Возвращаем нормализованные значения
	return train_norm, test_norm

