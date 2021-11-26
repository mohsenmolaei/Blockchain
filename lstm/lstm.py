# -*- coding: utf-8 -*-
#%%
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from tensorflow.keras import optimizers
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
import tensorflow as tf
# from google.colab import files
# files.upload()

# from IPython.display import display, HTML
# display(HTML(df.to_html()))
#%%
class DataProcessing:
    def __init__(self,feature_name,inputs, train):
        self.feature_name=feature_name
        self.train = train
        self.inputs = inputs
        self.i = int(self.train * len(self.inputs))
        self.stock_train = self.inputs[0: self.i]
        self.stock_test = self.inputs[self.i:]
        self.input_train = []
        self.output_train = []
        self.input_test = []
        self.output_test = []

    def gen_train(self, seq_len):
        """
        Generates training data
        :param seq_len: length of window
        :return: X_train and Y_train
        """
        for i in range((len(self.stock_train[self.feature_name])//seq_len)*seq_len - seq_len):
            x = np.array(self.stock_train[self.feature_name].iloc[i: i + seq_len])
            y = np.array([self.stock_train['price'].iloc[i + seq_len ]], np.float64)
            self.input_train.append(x)
            self.output_train.append(y)
        self.X_train = np.array(self.input_train)
        self.Y_train = np.array(self.output_train)

    def gen_test(self, seq_len):
        """
        Generates test data
        :param seq_len: Length of window
        :return: X_test and Y_test
        """
        for i in range((len(self.stock_test[self.feature_name])//seq_len)*seq_len - seq_len ):
            x = np.array(self.stock_test[self.feature_name].iloc[i: i + seq_len])
            y = np.array([self.stock_test['price'].iloc[i + seq_len ]], np.float64)
            self.input_test.append(x)
            self.output_test.append(y)
        self.X_test = np.array(self.input_test)
        self.Y_test = np.array(self.output_test)
#%%

feature_name='price' ##########################
seq=10
#get data
df = pd.read_csv("top_features.csv",index_col=None)
inputs=np.array([])
#reshape
X = np.array(df['price']).reshape(-1,1) 
scaler_test = MinMaxScaler()
scaler_test.fit(X)
X_scaled = scaler_test.transform(X)
price = X_scaled.reshape(1,-1)[0]
#diff

X2 = np.diff(np.log(np.array(df[feature_name]))).reshape(-1,1) 
# X2 = np.array(df[feature_name]).reshape(-1,1) 
scaler = MinMaxScaler()
scaler.fit(X2)
X_scaled = scaler.transform(X2)
feature_X = X_scaled.reshape(1,-1)[0]
#concat
feature_name='lable'
inp=pd.concat([pd.DataFrame(price),pd.DataFrame(feature_X)],axis=1)
inputs=pd.DataFrame(data=inp)
inputs.columns =['price', feature_name]

#DataProcessing
process = DataProcessing(feature_name,inputs, 0.80)
process.gen_test(seq)
process.gen_train(seq)

X_train = process.X_train.reshape((process.X_train.shape[0],seq,1))
Y_train = process.Y_train

X_test = process.X_test.reshape(process.X_test.shape[0],seq,1)
Y_test = process.Y_test

#%%
np.random.seed(1234)
tf.random.set_seed(1234)

regressor = Sequential()
# LSTM layer 
regressor.add(LSTM(units = 32, activation = 'relu', return_sequences = True, input_shape = (seq, 1)))
regressor.add(LSTM(units = 32, activation = 'relu'))
# Fully connected layer
regressor.add(Dense(units = 1))
regressor.summary()

# Compiling the RNN
regressor.compile(optimizers.Adam(learning_rate=1e-3), loss = 'mean_squared_error')
#Fitting the RNN model
history= regressor.fit(X_train, Y_train, epochs = 150, batch_size=16, validation_split=0.15, shuffle=True )

#%%
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(len(loss))
plt.figure()
plt.plot(epochs, loss, 'b', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title("Training and Validation Loss")
plt.legend()
plt.show()
#%%
Y_pred  = regressor.predict(X_test)
# scale= 1/scaler_test.scale_
Y_test2 = scaler_test.inverse_transform(Y_test)
Y_pred2 = scaler_test.inverse_transform(Y_pred)

# print(scale)
plt.figure(figsize=(14,5))
plt.plot(Y_test2, color = 'red', label = 'Real Bitcoin Price')
plt.plot(Y_pred2, color = 'green', label = 'Predicted Bitcoin Price')

plt.title('Bitcoin Price Prediction using RNN-LSTM')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()

#%%
# print(regressor.layers[0].trainable_weights)
# regressor.save('model')
# regressor = tf.keras.models.load_model('model')
