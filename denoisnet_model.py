import tensorflow as tf
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, UpSampling1D, concatenate

def denoisnet_model(input_shape):
    inputs = Input(shape=input_shape)

    # Encoder
    conv1 = Conv1D(64, 3, activation='relu', padding='same')(inputs)
    pool1 = MaxPooling1D(2)(conv1)
    
    conv2 = Conv1D(128, 3, activation='relu', padding='same')(pool1)
    pool2 = MaxPooling1D(2)(conv2)
    
    conv3 = Conv1D(256, 3, activation='relu', padding='same')(pool2)
    pool3 = MaxPooling1D(2)(conv3)
    
    # Bottleneck
    conv4 = Conv1D(512, 3, activation='relu', padding='same')(pool3)
    
    # Decoder
    up1 = UpSampling1D(2)(conv4)
    concat1 = concatenate([up1, conv3])
    conv5 = Conv1D(256, 3, activation='relu', padding='same')(concat1)
    
    up2 = UpSampling1D(2)(conv5)
    concat2 = concatenate([up2, conv2])
    conv6 = Conv1D(128, 3, activation='relu', padding='same')(concat2)
    
    up3 = UpSampling1D(2)(conv6)
    concat3 = concatenate([up3, conv1])
    conv7 = Conv1D(64, 3, activation='relu', padding='same')(concat3)
    
    # Output
    outputs = Conv1D(input_shape[-1], 3, activation='sigmoid', padding='same')(conv7)
    
    # Create model
    model = tf.keras.Model(inputs, outputs)
    return model