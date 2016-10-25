## Multiclass Classification with the Hidden Preferred dataset
import numpy
import pandas
from keras.models import Sequential
from keras.models import model_from_json
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers.normalization import BatchNormalization
from keras.wrappers.scikit_learn import KerasClassifier
from keras.optimizers import SGD
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
#from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline

## Set random seed for reproducibility
seed = 7
numpy.random.seed(seed)

## Load training/test and production datasets
dataframe = pandas.read_csv("hp500.csv", delimiter=",", header=None)
dataset = dataframe.values
X = dataset[:,1:100].astype(float)
Y = dataset[:,100]
dataframe2 = pandas.read_csv("hp1000.csv", delimiter=",", header=None)
dataset2 = dataframe2.values
X2 = dataset2[:,1:100].astype(float)

## Encode class values as integers
encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)

## Convert integers to dummy variables (ie. one-hot encoded)
dummy_y = np_utils.to_categorical(encoded_Y)

## Define Deep Neural Network model
def dnn_model():
	# Create model
	model = Sequential()
#	model.add(Dropout(0.2, input_shape=(99,)))
	model.add(Dense(100, input_dim=99, init='normal', activation='relu'))
#	model.add(BatchNormalization(epsilon=1e-05, mode=0, axis=-1, momentum=0.99, weights=None, beta_init='zero', gamma_init='one', gamma_regularizer=None, beta_regularizer=None))
	model.add(BatchNormalization(epsilon=1e-05, mode=0, axis=-1, momentum=0.99, weights=None, beta_init='zero', gamma_init='one'))
	model.add(Dropout(0.2))
        model.add(Dense(250, input_dim=250, init='normal', activation='relu'))
        model.add(Dense(250, input_dim=250, init='normal', activation='relu'))
        model.add(Dense(250, input_dim=250, init='normal', activation='relu'))
	model.add(Dense(2, init='normal', activation='sigmoid'))
	# Compile model
	epochs = 50
	learning_rate = 0.1
	decay_rate = learning_rate / epochs
	momentum = 0.8
#	sgd = SGD(lr=0.1, momentum=0.9, decay=0.0, nesterov=False)
	sgd = SGD(lr=learning_rate, momentum=momentum, decay=decay_rate, nesterov=False)
#	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
	return model

## Build, train, validate/test and run predictions on Deep Neural Network model
deepnn = KerasClassifier(build_fn=dnn_model, nb_epoch=200, batch_size=5, verbose=0)

#kfold = KFold(n=len(X), n_folds=10, shuffle=True, random_state=seed)
#results = cross_val_score(deepnn, X, dummy_y, cv=kfold)
#print("CV Accuracy: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))

X_train, X_test, Y_train, Y_test = train_test_split(X, dummy_y, test_size=0.33, random_state=seed)
deepnn.fit(X_train, Y_train)
#predictions_class = deepnn.predict(X_test, batch_size=32, verbose=0)
score = deepnn.score(X_train, Y_train, batch_size=32)
predictions_class = deepnn.predict(X2)
predictions_prob = deepnn.predict_proba(X2)
print("First 25 predictions: %s" % (predictions_class[:25]))
print("Predictions:")
print(predictions_class)
print(predictions_prob)
#print(encoder.inverse_transform(predictions_class))
predictions_out = encoder.inverse_transform(predictions_class)
#predictions_out = (predictions_prob)

## Output predictions to a file
counter = 0
f = open("predictions_dnn1.tmp", mode="w")
for item in predictions_out:
	counter += 1
	f.write(str(counter) + "," + str(item) + "\n")
f.close()

counter = 0
f = open("predictions_dnn2.tmp", mode="w")
for line in open("hp1000.csv"):
	columns = line.split(",")
	counter += 1
	f.write(str(counter) + "," + str(columns[0]) + "\n")
f.close()

a_cols = ['id', 'prediction']
a = pandas.read_csv('predictions_dnn1.tmp', sep=',', names=a_cols, encoding='latin-1')
b_cols = ['id', 'cardno']
b = pandas.read_csv('predictions_dnn2.tmp', sep=',', names=b_cols, encoding='latin-1')
merged = b.merge(a, on='id')
merged.to_csv("predictions_dnn.txt", index=False)

## Ensembling Dnn and Xgb predictions to a file
a_cols = ['id','cardno','pred_dnn']
a = pandas.read_csv('predictions_dnn.txt', sep=',', names=a_cols, encoding='latin-1')
b_cols = ['id','cardno','pred_xgb']
b = pandas.read_csv('predictions_xgb.txt', sep=',', names=b_cols, encoding='latin-1')
merged = b.merge(a, on=('id', 'cardno'))
merged.to_csv("predictions.txt", index=False)
#c_cols = ['id','cardno','pred1','pred2']
#c = pandas.read_csv('predictions.txt', sep=',', names=a_cols, encoding='latin-1')

## Serialize model to JSON
#model_json = deepnn.to_json()
#with open("dnn_model.json", "w") as json_file:
#	json_file.write(model_json)
## Serialize weights to HDF5
#deepnn.save_weights("dnn_model.h5")
#print("Saved DeepNN model and weights to file")


