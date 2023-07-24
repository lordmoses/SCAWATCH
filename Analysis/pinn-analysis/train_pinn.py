# %%
import numpy as np
import pandas as pd
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from vae_methods import *
# from fpfn import function
import seaborn as sns
from IPython import display
import matplotlib.pyplot as plt
import math


# # Reading in the data. 
# ## For training, we combined several files which had no attacks and for testing we used a single file where there were attacks and no attacks.
# ## The training data has 157579 datapoints and testing data has 1924 datapoints, out of which 956 are instances without attacks and 968 are instances with attacks

train = pd.read_csv('/home/user/Transformer-based_physics_ae/data/Combined.csv')
train.drop(["millis"], axis = 1, inplace = True)

test = pd.read_csv('/home/user/Transformer-based_physics_ae/data/chemical_dosing_attack_normal_time_series_data.csv')
test.drop(['millis'], axis = 1, inplace = True)

# ## Our training dataframe doesn't have the label column. However, since we know that all datapoints are instances without attacks we set the label of all the datapoints as 0

train['attack'] = train.shape[0] * [0]

# ## Several columns in our data are static, i.e. the values in the column remain the same throughout the entire data. We will drop these static columns to reduce dimensionality.  
# ## We also try to find categorical columns, so we can apply one-hot encoding to them. All other columns are numerical, so we apply scaling to them.


for cols in train.columns:
    print(cols)
    print("Min:",train[cols].min())
    print("Max:",train[cols].max())
    print()

# ## We find static columns by finding the columns whose maximum value is equal to the minimum value. There are 13 such static columns other than our label column.
# ## Categorical columns are found by checking number of unique values in a column. If the number of unique values is between 2 and a threshold (we set it as 5) we consider that column as categorical, if the number of unique values are greater than 5, then we consider it as a numerical column. We found no categorical columns in this dataset.


static_columns = []

for column in train.columns:
    if train[column].max() == train[column].min():
        static_columns.append(column)
        
def check_categorical(dataframe, threshold):
    
    categorical_columns = {}

    for cols in dataframe.columns:
        
        if dataframe[cols].nunique() in range(2,threshold):
            
            categorical_columns[cols] = dataframe[cols].nunique()

    return categorical_columns


categorical_columns = check_categorical(train,5)



label_column = ['attack']

numerical_columns = list(set(train.columns) - set(static_columns + label_column))

## In the preprocessing step we drop the static columns and label column. We also apply scaling and return the scaler, so it can be used on the testing data.

def preprocess(dataframe, static_columns, label_column, numerical_columns, scaler = None):
    
    try: 
        df = dataframe.drop(columns = static_columns + label_column)
    except KeyError:
        df = dataframe.drop(columns = static_columns)
        

    labels = dataframe[label_column]
    
    if scaler:
        df = scaler.transform(df)
        return df,labels
    else:
        scaler = StandardScaler()
        df = scaler.fit_transform(df)
        return df,labels,scaler
    

# ## We convert our data to sequences, since time-series models including transformers work on sequences.
# ## Suppose our data is [1, 2, 3, 4, 5] and the labels are [0, 0, 1, 1, 1], if we choose a sequence size of 2, our training data will be converted into this form:
# ## X = [[1, 2], [2, 3], [3, 4]], y = [1, 1, 1] where x is the data and y are the labels


def to_sequences(x, y, seq_size = 1):
    
    x_vals = []
    y_vals = []
    
    for i in range(len(x) - seq_size):
        x_vals.append(x[i : (i+seq_size)])
        y_vals.append(y.values[i+seq_size])              
        
    return np.array(x_vals), np.array(y_vals)


x_train, y_train, scaler = preprocess(train, static_columns, label_column, numerical_columns)
x_test, y_test = preprocess(test, static_columns, label_column, numerical_columns, scaler)



seq_size = 4
x_train_seq, y_train_seq = to_sequences(x_train, y_train, seq_size)
x_test_seq, y_test_seq = to_sequences(x_test, y_test, seq_size) 



def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    # Normalization and Attention
    x = layers.LayerNormalization(epsilon=1e-6)(inputs)
    x = layers.MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=dropout
    )(x, x)
    x = layers.Dropout(dropout)(x)
    res = x + inputs

    # Feed Forward Part
    x = layers.LayerNormalization(epsilon=1e-6)(res)
    x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    return x + res

input_shape = x_train_seq.shape[1:]
features = x_train_seq.shape[-1]


# %%
## Formula used: T = I * d^2θ/dt^2

## Function to calculate torque and moment of Inertia, given our data and moment of inertia

def calculate_new_physics(data):

    d2xdt2 = np.gradient(np.gradient(data))
    cos_d2xdt2 = tf.math.cos(d2xdt2)

    v = np.gradient(data)
    sin_dvdt = tf.math.sin(np.gradient(v))

    return sin_dvdt, cos_d2xdt2 

# %% [markdown]
# To implement attention in our VAE, we first put in some transformer blocks, after which we apply pooling. 
# Applying pooling helps us to feed the output of the transformer blocks to the fully connected layers. After pooling we have 2 dense layers. This makes up our encoder half. The output of the encoder is mapped into a latent space (bottleneck layer). The output of the latent space is then passed through a repeat vector layer, which just copies the input and concatenates it to itself to increase dimensionality. This is then passed again to transformer blocks. The output of the transformer blocks is passed through 2 Fully connected layers, then we finally arrive at the output layer

def attention_vae(latent_dim,
    input_shape,
    head_size,
    num_heads,
    ff_dim,
    num_transformer_blocks,
    pinn_loss_weight):
    
    inputs = keras.Input(shape = input_shape)
    x = inputs
    
    for _ in range(num_transformer_blocks):
        x = transformer_encoder(x, head_size, num_heads, ff_dim)
        
    x = layers.GlobalAveragePooling1D(data_format="channels_first")(x)
    
    dense1_vae = layers.Dense(features - 3, activation='relu')(x)

    dense2_vae = layers.Dense(features - 5, activation='relu')(dense1_vae)

    z_mean = layers.Dense(latent_dim, name="z_mean")(dense2_vae)
    z_log_var = layers.Dense(latent_dim, name="z_log_var")(dense2_vae)
    z = Sampling()([z_mean, z_log_var])

    encoder_vae = keras.Model(inputs=inputs,
                            outputs=[z_mean, z_log_var, z],
                            name='LSTM_VAE_encoder')

    inputs_decoder_vae = keras.Input(shape=(latent_dim,))

    repeat_vec = layers.RepeatVector(seq_size)(inputs_decoder_vae)

    for i in range(num_transformer_blocks):
        if i == 0:
            x = transformer_encoder(repeat_vec, head_size, num_heads, ff_dim)
        else:
            x = transformer_encoder(x, head_size, num_heads, ff_dim)
            

    dense1_decoder_vae = layers.Dense(features - 3, activation='relu')(x)

    dense2_decoder_vae = layers.Dense(features - 1, activation='relu')(dense1_decoder_vae)

    decoder_outputs_vae = layers.Dense(features, name='decoder_outputs')(dense2_decoder_vae)

    decoder_vae = keras.Model(inputs=inputs_decoder_vae,
                            outputs=decoder_outputs_vae,
                            name='attention_VAE_decoder')

    attention_vae = VAE(encoder_vae, decoder_vae)

    def vae_loss(data, reconstruction): #y_true, y_pred
       
        mu, ln_var, z = attention_vae.encoder(data)
        reconstruction_loss = tf.reduce_mean(
            keras.losses.mean_squared_error(data, reconstruction)
        )
        kl_loss = 1 + ln_var - tf.square(mu) - tf.exp(ln_var)
        kl_loss = tf.reduce_mean(kl_loss)
        kl_loss *= -0.5
        
        # Physics-informed loss
        sin_dvdt, _ = calculate_new_physics(reconstruction)
        _ , cos_d2xt2,  = calculate_new_physics(data)

        inertia_delay = np.random.uniform(3.5,5.5)

        physics_loss = tf.reduce_mean(keras.losses.mean_squared_error(
            sin_dvdt * inertia_delay, cos_d2xt2))
        

        # tf.print(physics_loss)

        # Total loss
        total_loss = (1-pinn_loss_weight) * (reconstruction_loss + kl_loss) + (pinn_loss_weight * physics_loss)
        
        return total_loss

    attention_vae.compile(loss=vae_loss, optimizer=keras.optimizers.Adam(learning_rate=0.001, clipnorm=1), run_eagerly = True)

    return attention_vae
    
    

# %% [markdown]
# ## We define two callbacks. Early stopping and reduced learning rate. Reduced learning rate allows us to reduce the learning rate if the validation loss isn't changing for 5 epochs early stopping stops the training of the model if the validation loss doesn't improve for 10 epochs. The best weights are saved.

# %%
early_stopping_callback_hyperopt = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=50,
    restore_best_weights=True,
)


reduce_lr_calllback = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.15,
    patience=5,
    verbose=0,
    mode='auto',
    min_delta=0.0001,
    cooldown=0,
    min_lr=0
)

path_checkpoint = "/home/user/Transformer-based_physics_ae/code/seq_size4_1000eps/Checkpoint-{epoch:02d}.hdf5"

ckpt_callback = tf.keras.callbacks.ModelCheckpoint(filepath=path_checkpoint,
                                                 save_weights_only=True,
                                                 monitor='val_loss',
                                                 mode = 'min',
                                                 save_best_only=True)

logger = tf.keras.callbacks.CSVLogger('/home/user/Transformer-based_physics_ae/code/pinn_logs_pw03_seqsize4-1000eps.csv', separator=",", append=True)


attention_vae = attention_vae(latent_dim = features - 4,
                 input_shape = input_shape, 
                 head_size = 256,
                 num_heads = 8,
                 ff_dim = 4,
                 num_transformer_blocks = 8,
                 pinn_loss_weight = 0.3)


hist = attention_vae.fit(x_train_seq, x_train_seq,
                  epochs = 1000,
                  batch_size = 1028,
                  validation_split = 0.15,
                  callbacks = [reduce_lr_calllback, ckpt_callback, logger])  


train_reconstructions = attention_vae.predict(x_train_seq)
train_loss = keras.losses.mean_squared_error(x_train_seq, train_reconstructions)
threshold = np.quantile(train_loss, 0.14)


# ## To get predictions, we compute the loss of the data with the reconstruction(output of the model). If the loss is greater than the threshold, then the model hasn't reconstructed the input very well. Since our training data is comprised of only non-attack data, the reconstructions should be less than the threshold, if the input is non-attack data.
# ## So, if the loss is greater than the threshold, the input data must correspond to attack data. Else, it must correspond to non-attack data.


def predict(data, reconstructions, threshold):
    loss = tf.keras.losses.mean_squared_error(data, reconstructions)
    return ~tf.math.less(loss, threshold)

train_predictions = predict(x_train_seq, train_reconstructions, threshold)


## Since the predictions are of the same dimensions as of the input, we must convert them into (n * 1) dimension. We do this using numpy strided techniques.


strided_train_predictions = np.lib.stride_tricks.as_strided(train_predictions, shape=(train_predictions.shape[0],1), strides=np.array(train_predictions).strides)

## Result analysis

def show_results(model, x, labels, data_name, threshold, sequence = False, window = 2):
    
    if not sequence:
        data_seq = sequencify(x, window)
        reconstructions = model.predict(data_seq)
        sequencified_labels = labels[window-1:].reshape(-1,1)
        predictions = predict(data_seq, reconstructions, threshold)
        strided_predictions = np.lib.stride_tricks.as_strided(predictions, shape=(predictions.shape[0],1), strides=np.array(predictions).strides)
    else:
        reconstructions = model.predict(x)
        sequencified_labels = labels
        predictions = predict(x, reconstructions, threshold)
        strided_predictions = np.lib.stride_tricks.as_strided(predictions, shape=(predictions.shape[0],1), strides=np.array(predictions).strides)

    print("----------------------------")
    print(f"{data_name}ing Result")
    print("----------------------------")
    print("Accuracy = {}".format(accuracy_score(sequencified_labels, strided_predictions)))
    print("Precision = {}".format(precision_score(sequencified_labels, strided_predictions, zero_division=0)))
    print("Recall = {}".format(recall_score(sequencified_labels, strided_predictions, zero_division=0)))
    print("F1 = {}".format(f1_score(sequencified_labels, strided_predictions, zero_division=0)))

    
    cf = confusion_matrix(sequencified_labels,strided_predictions)
    ax = sns.heatmap(cf, annot=True, cmap='Blues')
    ax.set_title('Seaborn Confusion Matrix with labels\n\n');
    ax.set_xlabel('\nPredicted Values')
    ax.set_ylabel('Actual Values ');
    ## Ticket labels - List must be in alphabetical order
    ax.xaxis.set_ticklabels(['False','True'])
    ax.yaxis.set_ticklabels(['False','True'])
        

print("RESULTS ON TRAINING DATA: ")
show_results(attention_vae, x_train_seq, y_train_seq, "Train", threshold, sequence = True, window = seq_size)

print("RESULTS ON TESTING DATA: ")
show_results(attention_vae, x_test_seq, y_test_seq, "Test", threshold, sequence = True, window = seq_size)
