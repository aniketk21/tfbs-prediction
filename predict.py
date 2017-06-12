import numpy as np
import keras
from keras.models import load_model

model = load_model('model_myc.h5')

x = []
x.append([40742400, 40742600, 1, 0, 1000])
x.append([50000000, 51000000, 1, 0, 0])
for i in range(len(x)):
    x[i] = np.array(x[i])
    x[i] = x[i].reshape(1, 5)
    x[i] = x[i].astype('float32')

#op = model.predict(x, batch_size=32, verbose=1)
op = model.predict_on_batch(x)

print op
