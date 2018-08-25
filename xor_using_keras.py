import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense

# Traing Data
input_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], "float32")

output_data = np.array([[0], [1], [1], [0]], "float32")

# Setting up the model
model = Sequential()
model.add(Dense(16, input_dim=2, activation='sigmoid'))
model.add(Dense(1, activation='sigmoid'))

# Training the model
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['binary_accuracy'])
model.fit(input_data, output_data, nb_epoch=500, verbose=2)

# Testing the model
print(model.predict(input_data).round())
