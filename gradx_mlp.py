import numpy as np
import sys
import os

if len(sys.argv) < 2:
	print('python gradx_mlp dropoutrate')
	sys.exit()
else:
	p=float(sys.argv[1])

np.random.seed(1337)  # for reproducibility

X_train = np.random.rand(60000,3)
X_test =  np.random.rand(20,3)
y_train = np.zeros((60000,2))
y_test =  np.zeros((20,2))
y_train[:,0] = X_train[:,1]- X_train[:,0]
y_train[:,1] = X_train[:,2]- X_train[:,1]
y_test[:,0] = X_test[:,1]- X_test[:,0]
y_test[:,1] = X_test[:,2]- X_test[:,1]

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils

model = Sequential()
model.add(Dense(2, input_shape=(3,)))
model.add(Activation('linear'))
model.add(Dropout(p))

model.compile(loss='mse',
              optimizer='adadelta',
              metrics=['accuracy'])

batch_size = 100
nb_epoch = 10

model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=nb_epoch,
          verbose=1, validation_data=(X_test, y_test))
score = model.evaluate(X_test, y_test, verbose=1)
result=model.predict(X_test, verbose=1)

print('Test score:', score[0])
print('Test accuracy:', score[1])
print('Result:')
print(result)
print('Dropout:', p)
print('Weight is rescaled to ', 1-p)
print('get_weights WB:')
WB=model.get_weights()
print('weight W:')
print(WB[0])
print('bias B:')
print(WB[1])

'''
# write
json_string = model.to_json()
open('gradx_mlp.json', 'w').write(json_string)
model.save_weights('gradx_mlp_weights.h5')
# read
from keras.models import model_from_json
model = model_from_json(open('gradx_mlp.json').read())
os.system('set HDF5_DISABLE_VERSION_CHECK=1')
WB=model.load_weights('gradx_mlp_weights.h5')
'''

print('==========Our gradx_mlp==========')
WB=model.get_weights()
W=WB[0]
B=WB[1]
for ioutput in range(2):
	for iinput in range(3):
		print('W[{}][{}]: input{} --> output{}: {:6.2f} + bias{}: {:6.2f}'.format(iinput, ioutput, iinput, ioutput, W[iinput][ioutput], ioutput, B[ioutput]))
