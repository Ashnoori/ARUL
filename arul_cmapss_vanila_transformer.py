#!/usr/bin/env python
# coding: utf-8

# # Objective:
# The turbofan dataset features four datasets of increasing complexity. The engines operate normally in the beginning but develop a fault over time. For the training sets, the engines are run to failure, while in the test sets the time series end ‘sometime’ before failure. The goal is to predict the Remaining Useful Life (RUL) of each turbofan engine.
# 
# The engine runs till a specified point in the test dataset. The goal is predicting the RUL at the last point of dataset.
# 
# The datasets comprise simulations of several turbofan engines across time, with each row including the following information:
# 1.   Engine unit number
# 2.   Time, in cycles
# 1.   Three operational settings
# 2.   21 sensor readings

# In[ ]:


# Loading the data: just upload the data trian_XXX.txt 
#to your Google Drive and update the path in line 6.


from google.colab import drive
drive.mount("/content/drive")


# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import sklearn.metrics
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense, Dropout
import tensorflow as tf
import math
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from tensorflow.keras.optimizers import Adam


# In[ ]:


# RUL_XXX.txt path
rul_1_gt = pd.read_csv("/content/drive/My Drive/data/cmapss/RUL_FD001.txt", header=None)

rul_1_gt.rename(columns={0: "RUL_1gt"}, inplace=True)   

column_name_dict={ 0: "engine_id", 1: "cycle", 2: "op_set_1", 3: "op_set_2", 4: "op_set_3", 5:"sensor_1", 6: "sensor_2",
                   7: "sensor_3", 8: "sensor_4", 9: "sensor_5", 10: "sensor_6", 11: "sensor_7", 12: "sensor_8", 13: "sensor_9",
                  14: "sensor_10", 15: "sensor_11", 16: "sensor_12", 17: "sensor_13", 18: "sensor_14", 19: "sensor_15", 20: "sensor_16",
                  21: "sensor_17", 22: "sensor_18", 23: "sensor_19", 24: "sensor_20", 25: "sensor_21", 26: "sensor_22", 27: "sensor_23"}

################   ################   ################   ################ 
# test_xxx.txt path
test_1 = pd.read_csv("/content/drive/My Drive/data/cmapss/test_FD001.txt", header=None, sep=' ')

test_1.rename(columns=column_name_dict, inplace=True)

################   ################   ################   ################ 
#train_xxx.txt path
train_1 = pd.read_csv("/content/drive/My Drive/data/cmapss/train_FD001.txt", header=None, sep=' ')

train_1.rename(columns=column_name_dict, inplace=True)


# In[ ]:


# Feture selection
train_1.drop(
    columns=["op_set_1","op_set_2","op_set_3",
             "sensor_1","sensor_5","sensor_6",
             "sensor_10","sensor_16","sensor_18",
             "sensor_19","sensor_22","sensor_23"], inplace=True)


test_1.drop(
    columns=["op_set_1","op_set_2","op_set_3",
             "sensor_1","sensor_5","sensor_6",
             "sensor_10","sensor_16","sensor_18",
             "sensor_19","sensor_22","sensor_23"], inplace=True)


# In[ ]:


train_1


# In[ ]:


test_1


# In[ ]:


#Compute RUL values -- RUL for train_1 
rul_1 = pd.DataFrame(train_1.groupby('engine_id')['cycle'].max()).reset_index()
rul_1.columns = ['engine_id', 'max']

rul_train_1 = train_1.merge(rul_1, on=['engine_id'], how='left')
rul_train_1['RUL'] = rul_train_1['max'] - rul_train_1['cycle']
rul_train_1.drop(['max'], axis=1, inplace=True)


# In[ ]:


rul_train_1


# In[ ]:


#Compute RUL values -- RUL for test_1 
rul_1_gt["engine_id"]=rul_1_gt.index + 1

max_1 = pd.DataFrame(test_1.groupby('engine_id')['cycle'].max()).reset_index()
max_1.columns = ['engine_id', 'max']
max_test_1 = test_1.merge(max_1, on=['engine_id'], how='left')
rul_test_1 = max_test_1.merge(rul_1_gt, on=['engine_id'], how='left')

rul_test_1['RUL'] = rul_test_1['max'] - rul_test_1['cycle'] + rul_test_1["RUL_1gt"] 
rul_test_1.drop(['max', 'RUL_1gt'], axis=1, inplace=True)


# In[ ]:


rul_test_1


# In[ ]:


df_max_rul = rul_train_1[['engine_id', 'RUL']].groupby('engine_id').max().reset_index()
df_max_rul['RUL'].hist(bins=15, figsize=(15,7))
plt.xlabel('RUL')
plt.ylabel('frequency')
plt.show()


# The histogram reconfirms most engines break down around 200 cycles. Furthermore, the distribution is right skewed, with few engines lasting over 300 cycles.

# In[ ]:


#heatmap for correlation coefficient
df_corr = rul_train_1.drop(columns=["engine_id"]).corr()
sns.set(font_scale=0.9)
plt.figure(figsize=(24,16))
sns.heatmap(df_corr, annot=True, fmt=".4f",vmin=-1, vmax=1, linewidths=.5, cmap = sns.color_palette("ch:s=.25,rot=-.25", 200))

plt.figtext(.6, 0.9,'correlation matrix of train_1', fontsize=30, ha='center')
plt.xticks(rotation=90)
plt.show()


# In[ ]:


# Creating piece-wise data frame from rul_train_1
pw_train_1 = rul_train_1.copy()
pw_train_1['RUL'] = np.where((pw_train_1.RUL > 125), 125, pw_train_1.RUL)
pw_train_1


# In[ ]:


# Creating piece-wise data frame from rul_test_1
pw_test_1 = rul_test_1.copy()
pw_test_1['RUL'] = np.where((pw_test_1.RUL > 125), 125, pw_test_1.RUL)
pw_test_1


# In[ ]:


rul_train_1


# In[ ]:


rul_test_1


# In[ ]:


# Just for plotting RUL
pw = rul_train_1.sort_values(by=['RUL'])#, ascending=False
pw['RUL'] = np.where((rul_train_1.RUL > 125), 125, rul_train_1.RUL)
pw = pw.sort_values(by = 'RUL', ascending=False)

# Plotting RUL
ax = plt.gca()
pw.plot(kind='line',x='engine_id',y='RUL',use_index=False, ax=ax)
plt.show()


# In[ ]:


# max lifetime for each engine - Linear degredation function
train_1.groupby('engine_id')['cycle'].max()


# In[ ]:


# max lifetime for each engine - Picewise Linear degredation function
pw_train_1.groupby('engine_id')['RUL'].max()


# In[ ]:


# train_1 minmax scaling
cols_normalize_1 = pw_train_1.columns.difference(['engine_id','cycle'])

scaler_1 = MinMaxScaler()
norm_rul_train_1 = pd.DataFrame(scaler_1.fit_transform(pw_train_1[cols_normalize_1]), 
                                columns=cols_normalize_1, 
                                index=pw_train_1.index)

norm_rul_train_1=pd.concat([norm_rul_train_1, pw_train_1[["engine_id", "cycle"]]], axis=1)

################   ################   ################   ################   ################

# test_1 minmax scaling
norm_rul_test_1 = pd.DataFrame(scaler_1.transform(pw_test_1[cols_normalize_1]), 
                                columns=cols_normalize_1, 
                                index=pw_test_1.index)

norm_rul_test_1=pd.concat([norm_rul_test_1, pw_test_1[["engine_id", "cycle"]]], axis=1)


# In[ ]:


# Because we only have True RUL values for those records, we are only interested in the LAST CYCLE of each engine in the test set.
# sort lists by the # of engines
g_train_1=norm_rul_train_1.groupby('engine_id')
g_test_1=norm_rul_test_1.groupby('engine_id')

#list of dfs(engines)
train_list = []
test_list = []  

for engineid in g_train_1.groups.keys():
    train_list.append(g_train_1.get_group(engineid)) 

for engineid in g_test_1.groups.keys():
    test_list.append(g_test_1.get_group(engineid))


# In[ ]:


# creating sequences for each engine
from numpy import array

# df: df extracted from train_list, n_steps: window size
def split_sequences(df, n_steps):
    X, y = list(), list()
    for i in range(len(df)):
        end_ix = i + n_steps
        if end_ix > len(df):
            break
        seq_x, seq_y = df[i:end_ix, 1:], df[end_ix-1, 0]
        X.append(seq_x)
        y.append(seq_y)
    return array(X), array(y)


list_x=[]
list_y=[]
for engine_df in train_list:
    #convert df to arr 
    engine_arr=engine_df.drop(columns=["engine_id", "cycle"]).to_numpy()
    X, y = split_sequences(engine_arr, 21)#since smallest df has 21 rows
    list_x.append(X)
    list_y.append(y)

X_arr_train=np.concatenate(list_x)
y_arr_train=np.concatenate(list_y)


# In[ ]:


X_arr_train.shape


# In[ ]:


y_arr_train.shape


# In[ ]:


from tensorflow import keras
from tensorflow.keras import layers

class MultiHeadSelfAttention(layers.Layer):
    def __init__(self, embed_dim, num_heads=8):
        super(MultiHeadSelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embedding dimension = {embed_dim} should be divisible by number of heads = {num_heads}"
            )
        self.projection_dim = embed_dim // num_heads
        self.query_dense = layers.Dense(embed_dim)
        self.key_dense = layers.Dense(embed_dim)
        self.value_dense = layers.Dense(embed_dim)
        self.combine_heads = layers.Dense(embed_dim)

    def attention(self, query, key, value):
        score = tf.matmul(query, key, transpose_b=True)
        dim_key = tf.cast(tf.shape(key)[-1], tf.float32)
        scaled_score = score / tf.math.sqrt(dim_key)
        weights = tf.nn.softmax(scaled_score, axis=-1)
        output = tf.matmul(weights, value)
        return output, weights

    def separate_heads(self, x, batch_size):
        """Reshape x to the shape (batch_size, -1, num_heads, embedding dimension.
        Used to obtain the separate attention heads in each for each batch."""
        # -1 to ensure the dimension of the reshaped tensor is compatible with
        # the original x
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.projection_dim))
        # ensure the transposed tensor dimensions are 0, 2, 1, 3
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, inputs):
        # x.shape = [batch_size, seq_len, embedding_dim]
        batch_size = tf.shape(inputs)[0]
        query = self.query_dense(inputs)  # (batch_size, seq_len, embed_dim)
        key = self.key_dense(inputs)  # (batch_size, seq_len, embed_dim)
        value = self.value_dense(inputs)  # (batch_size, seq_len, embed_dim)
        query = self.separate_heads(
            query, batch_size
        )  # (batch_size, num_heads, seq_len, projection_dim)
        key = self.separate_heads(
            key, batch_size
        )  # (batch_size, num_heads, seq_len, projection_dim)
        value = self.separate_heads(
            value, batch_size
        )  # (batch_size, num_heads, seq_len, projection_dim)
        attention, weights = self.attention(query, key, value)
        attention = tf.transpose(
            attention, perm=[0, 2, 1, 3]
        )  # (batch_size, seq_len, num_heads, projection_dim)
        concat_attention = tf.reshape(
            attention, (batch_size, -1, self.embed_dim)
        )  # (batch_size, seq_len, embed_dim)
        # Recombine the heads
        output = self.combine_heads(
            concat_attention
        )  # (batch_size, seq_len, embed_dim)
        return output


# In[ ]:


class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = MultiHeadSelfAttention(embed_dim, num_heads)
        self.ffn = keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim),]
        )

        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training):
        # Multi-head attention
        attn_output = self.att(inputs)
        # Apply dropout
        attn_output = self.dropout1(attn_output, training=training)
        # Add and norm
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        # Pass through FFNN, dropout again
        ffn_output = self.dropout2(ffn_output, training=training)
        # Final add and norm
        return self.layernorm2(out1 + ffn_output)


# In[ ]:


class TransformerBlockWithLeakyRelu(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlockWithLeakyRelu, self).__init__()
        self.att = MultiHeadSelfAttention(embed_dim, num_heads)
        self.ffn = keras.Sequential(
            [layers.Dense(ff_dim, activation="gelu"), layers.Dense(embed_dim),]
        )
        self.leakyrelu = keras.Sequential(
            [layers.Dense(ff_dim, activation="LeakyReLU"), layers.Dense(embed_dim),]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training):
        # Multi-head attention
        attn_output = self.att(inputs)
        # Apply dropout
        attn_output = self.dropout1(attn_output, training=training)
        # Apply leakyrelu after each sublayer
        attn_output = self.leakyrelu(attn_output)
        # Add and norm
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        # Pass through FFNN, dropout again
        ffn_output = self.dropout2(ffn_output, training=training)
        # Apply leakyrelu
        ffn_output = self.leakyrelu(ffn_output)
        # Final add and norm
        return self.layernorm2(out1 + ffn_output)


# In[ ]:


class PositionalEmbedding(layers.Layer):
    """Add positional embedding"""
    def __init__(self, maxlen, embed_dim):
        super(PositionalEmbedding, self).__init__()
        self.maxlen = maxlen
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, x):
        maxlen = self.maxlen
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        return x + positions



class LinearLayer(layers.Layer):
    """LinearLayer"""
    def __init__(self, maxlen):
        super(LinearLayer, self).__init__()
        self.linear = layers.Dense(units=maxlen)

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        x = self.linear(x)
        return x
      
class Conv2D_layer(layers.Layer):
    """Convolutional Layer"""
    def __init__(self, filters=3, kernel_size=2, activation="relu"):
        super(Conv2D_layer, self).__init__()
        self.Conv2D = layers.Conv2D(filters=filters, kernel_size=kernel_size, activation=activation)
    def call(self, x):
        x = self.Conv2D(x)
        return x


# In[ ]:


seq_len = X_arr_train.shape[1]  
num_features = X_arr_train.shape[2]


embed_dim = 512  # Embedding size
num_heads = 2 # 1 - Number of attention heads
ff_dim = 512 # 3 - feed forward network hidden dimension - Hidden layer size in feed forward network inside transformer
droupout_rate = 0.3  # 4 - Droupoutrate

inputs = layers.Input(shape=(seq_len, num_features))
embedding_layer = LinearLayer(embed_dim)
positional_embeddings = PositionalEmbedding(maxlen=seq_len, embed_dim=embed_dim)


# 3. Building the network:
x = embedding_layer(inputs) # "Embedding dimension is the sequence length"
x = positional_embeddings(x) # adding positional embeddings

transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim, droupout_rate)
transformer_block2 = TransformerBlock(embed_dim, num_heads, ff_dim, droupout_rate)

# 2 - Number of Transformer_blocks
x = transformer_block(x)
x = transformer_block2(x)

x = layers.GlobalAveragePooling1D()(x)
x = layers.Dropout(droupout_rate)(x)

# 5 - Activation parameter
x = layers.Dense(64, activation="sigmoid")(x)
x = layers.Dropout(droupout_rate)(x)
x = layers.Dense(32, activation="sigmoid")(x)
x = layers.Dropout(droupout_rate)(x)

outputs = layers.Dense(1)(x)
model = keras.Model(inputs=inputs, outputs=outputs)
model.summary()


# In[ ]:


# root mean squared error (rmse) for regression (only for Keras tensors)
def rmse(y_true, y_pred):
    from keras import backend
    return backend.sqrt(backend.mean(backend.square(y_pred - y_true), axis=-1))

# 6 - learning rate - determine the default value
model.compile(optimizer=Adam(learning_rate=0.001), loss=rmse)

#Callback in the validation loss for 20 consecutive epochs. 
callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=20)

# 7 - batch_size
# 8 - epochs
history = model.fit(
    X_arr_train, y_arr_train, batch_size=128, epochs=50, callbacks=[callback], validation_split=0.3)


# In[ ]:


# plotting training loss and validation loss
from matplotlib.pyplot import figure
figure(figsize=(10, 6), dpi=100)

plt.plot(history.history['loss'], 'o-',label='Training loss')
plt.plot(history.history['val_loss'], 'o-',label='Validation loss')
plt.legend()
plt.title('Loss Error')
plt.xlabel('Epochs')
plt.ylabel('RMSE')


# In[ ]:


#prepare Test set to make predictions
list_x_test=[]
list_y_test=[]

for engine_df in test_list:
    #convert df to arr 
    engine_arr=engine_df.drop(columns=["engine_id", "cycle"]).to_numpy()
    X, y = split_sequences(engine_arr, seq_len)
    list_x_test.append(X[-1].reshape((1, seq_len, num_features)))
    list_y_test.append(y[-1].reshape((1, )))  
Xx_arr_test=np.concatenate(list_x_test)
yy_arr_test=np.concatenate(list_y_test)


# In[ ]:


Xx_arr_test.shape


# In[ ]:


yy_arr_test.shape


# In[ ]:


# inverse scaling for RUL column
#dummy to take inverse only on one col --> "y_pred"
def invTransform(scaler, y_pred, colNames):
    dummy = pd.DataFrame(np.zeros((len(y_pred), len(colNames))), columns=colNames)
    dummy["RUL"] = y_pred
    dummy = pd.DataFrame(scaler.inverse_transform(dummy), columns=colNames)
    return dummy["RUL"].values


# In[ ]:


# Prediction on Test set
y_test_pred = model.predict(Xx_arr_test, verbose=0)

# Inverse scaling for y_pred values
y_test_pred_inv=invTransform(scaler_1, y_test_pred, cols_normalize_1)

y_test_pred_reshaped=y_test_pred_inv.reshape((len(y_test_pred_inv, )))
y_test_pred_reshaped.shape


# In[ ]:


y_truth=rul_1_gt["RUL_1gt"].values
y_truth.shape


# In[ ]:


print("Predicted = \n", np.round(y_test_pred_reshaped, 2), "\n\n Actual = \n", y_truth)  


# In[ ]:


import math
y_actual = y_truth
y_predicted = y_test_pred_reshaped
 
MSE = np.square(np.subtract(y_actual,y_predicted)).mean() 
 
RMSE = math.sqrt(MSE)

print("Mean Square Error: ", MSE)
print("Root Mean Square Error: ",RMSE)


# In[ ]:


def score_func(y_true, y_pred):
    score_list = [
                  round(score(y_true,y_pred), 2), 
                  round(mean_absolute_error(y_true,y_pred), 2),
                  round(mean_squared_error(y_true,y_pred), 2) ** 0.5,
                  round(r2_score(y_true,y_pred), 2)
                  ]

    print(f' compatitive score: {score_list[0]}')
    print(f' mean absolute error: {score_list[1]}')
    print(f' root mean squared error: {score_list[2]}')
    print(f' R2 score: {score_list[3]}')
    
    return 


# In[ ]:


score_func(y_truth, y_test_pred_reshaped)


# In[ ]:


def score(y_true, y_pred, a1=10, a2=13):
    score = 0
    d = y_pred - y_true
    for i in d:
        if i >= 0 :
            score += math.exp(i/a2) - 1   
        else:
            score += math.exp(- i/a1) - 1
    return score


# In[ ]:


score(y_truth,y_test_pred_reshaped)


# In[ ]:




