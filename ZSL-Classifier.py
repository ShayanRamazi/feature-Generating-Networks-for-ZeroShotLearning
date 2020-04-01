
from readData import readForDap
import keras
from keras.layers import Input, Dense,Dropout
from keras.models import Sequential, Model
(_,x_train, y_train), (_,test_x, test_y) = readForDap()
# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train-1, 50)
test_y = keras.utils.to_categorical(test_y-1, 50)
model = Sequential()
model.add(Dense(50, activation='relu', input_shape=(2048,)))
model.add(Dropout(0.8))
model.add(Dense(50, activation='softmax'))

model.summary()

model.compile(loss='binary_crossentropy',
				  optimizer=keras.optimizers.Adam(lr=0.001, beta_1=0.5, beta_2=0.999, amsgrad=False),
				  metrics=['accuracy'])

history = model.fit(x_train, y_train,
                    batch_size=512,
                    epochs=15,
                    verbose=1,
                    validation_split=0.2)

score = model.evaluate(test_x, test_y, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
