from classifier import build_classificationLayer
from readData import readH5file2
import keras
from keras.losses import binary_crossentropy

(x_train, y_train, _), (test_x, test_y,_) = readH5file2()
# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train-1, 50)
test_y = keras.utils.to_categorical(test_y-1, 50)

model=build_classificationLayer()
model.compile(loss=binary_crossentropy,
				  optimizer=keras.optimizers.Adam(lr=0.001, beta_1=0.5, beta_2=0.999, amsgrad=False),
				  metrics=['accuracy'])

history = model.fit(x_train, y_train,
                    batch_size=512,
                    epochs=20,
                    verbose=1,
                    validation_split=0.3)

score = model.evaluate(test_x, test_y, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

model.save('./models/classifierLayer.h5');
