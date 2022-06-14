#=================================================
#_____________ Тренировка модели _________________
#=================================================


def train_and_evaluate(model, trainX, trainY, testX, testY, epochs, batch_size, model_name = 'model.h5'):
	history = model.fit(trainX, trainY, epochs = epochs, batch_size = batch_size, validation_data=(testX, testY), verbose=0)
	# Оценка модели
	_, acc = model.evaluate(testX, testY, verbose=0)
	model.save(model_name)
	return acc, history
