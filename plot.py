from keras.utils import plot_model
from models.cnn_models import *
model = new_model_1()
plot_model(model, to_file='model.png')

