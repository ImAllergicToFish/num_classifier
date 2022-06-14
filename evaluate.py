#=================================================
#_______ Оценка качества работы модели по  _______
#___ критерию k-кратной перекрестной валидации ___
#=================================================

from matplotlib import pyplot as plt

# Построения графика точности и ошибок
# для обученной модели
def model_diagnostic(history):
	
	# Построение графика ошибок
	plt.subplot(2, 1, 1)
	plt.title('Cross Entropy Loss')
	plt.plot(history.history['loss'], color='blue', label='train')
	plt.plot(history.history['val_loss'], color='orange', label='test')
	# Построение графика точности
	plt.subplot(2, 1, 2)    
	plt.title('Classification Accuracy')
	plt.plot(history.history['accuracy'], color='blue', label='train')
	plt.plot(history.history['val_accuracy'], color='orange', label='test')
	plt.show()

