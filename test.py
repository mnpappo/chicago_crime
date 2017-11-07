from keras.layers import Input, Dense
from keras.models import Model
from keras import optimizers

import numpy as np 
import pandas as pd 
from keras.utils import to_categorical

epochs = 2
batch_size = 4


crimes = pd.read_csv('./Chicago_Crimes_2012_to_2017.csv', error_bad_lines=False, usecols=["Primary_Type", "Ward"], index_col=None)

print('Dataset Shape before drop_duplicate : ', crimes.shape)
data = crimes.head(10000)


#### Input data ####
X = list(data.Primary_Type)

# define a mapping of chars to integers
char_to_int = dict((c, i) for i, c in enumerate(X))
int_to_char = dict((i, c) for i, c in enumerate(X))

# integer encode input data, just keeping the input data as int
X = np.array([char_to_int[char] for char in X], dtype=np.float32)
X = X.reshape((len(X), 1))


##### Labels #####
Y = np.array(list(data.Ward))
Y = to_categorical(Y)

output_shape = Y.shape[1]

# print (X)
# print (X.shape)
# print (Y)
# print (Y.shape)

# This returns a tensor
inputs = Input(shape=(1, ))
# a layer instance is callable on a tensor, and returns a tensor
x = Dense(32, activation='relu')(inputs)
x = Dense(32, activation='relu')(x)
x = Dense(64, activation='relu')(x)
x = Dense(64, activation='relu')(x)
predictions = Dense(output_shape, activation='softmax')(x)

# This creates a model that includes
# the Input layer and three Dense layers
model = Model(inputs=inputs, outputs=predictions)

sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

model.compile(optimizer=sgd, loss='mse', metrics=['accuracy'])
model.summary()
model.fit(X, Y, batch_size=batch_size, shuffle=True, epochs=epochs)  # starts training
model.save('model.h5')