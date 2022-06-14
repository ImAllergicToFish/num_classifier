from load_dataset import load_dataset
from normalize_img import normalize_pixels
from train_model import train_and_evaluate
from cnn_model import define_cnn_model
from evaluate import model_diagnostic
from test_model import load_image, predict
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

#=====================================================
#_________НАСТРОЙКА ПАРАМЕТРОВ ДЛЯ ОБУЧЕНИЯ___________
#=====================================================

EPOCHS = 10
BATCH_SIZE = 32
TRAIM_IMGS_QUANTITY = 60000 #MAX = 60000
TEST_IMGS_QUANTITY = 10000 #MAX = 10000

# Имя под которым будем сохранена модель
MODEL_NAME = 'model.h5'

# Для настройки модели нейронной сети см.: cnn_model.py

#======================================================

def main():

    # Загрузка датасета
    trainX, trainY, testX, testY = load_dataset()
    
    # Подготовка данных для обучения
    trainX, testX = normalize_pixels(trainX, testX)
	
    # Инициализация модели
    model = define_cnn_model()
    model.summary()

    # Обучение и оценка модели
    score, history = train_and_evaluate(
        model, 
        trainX = trainX[0:TRAIM_IMGS_QUANTITY], 
        trainY = trainY[0:TRAIM_IMGS_QUANTITY], 
        testX = testX[0:TEST_IMGS_QUANTITY], 
        testY = testY[0:TEST_IMGS_QUANTITY],
        batch_size= BATCH_SIZE,
        epochs = EPOCHS, 
        model_name = MODEL_NAME
    )
	
    model_diagnostic(history)
    print('SCORE:  %.3f' % (score * 100.0))
    

# Раскомментировать если нужно обучение

main()

#=====================================================
#_________________ТЕСТИРОВАНИЕ________________________

#/home/neleps/NEURAL_LABS/num_classifier/images
#img_path = './images/2.png'
#prediction = predict(MODEL_NAME, img_path)

fig = plt.figure()
ax = []
for i in range(10):
    img_path = './images/' + str(i) + '.png'
    prediction = predict(MODEL_NAME, img_path)
    image = mpimg.imread(img_path)
    
    ax.append(fig.add_subplot(2, 5, i+1))
    ax[i].title.set_text('Predict: ' + str(prediction))
    ax[i].imshow(image)

plt.show()


#print('PREDICTION: ' + str(prediction))