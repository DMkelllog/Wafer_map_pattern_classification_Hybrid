#!/usr/bin/env python
# coding: utf-8

# In[1]:


from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten,GlobalAveragePooling2D, concatenate
from tensorflow.keras import models
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Ridge
from sklearn.svm import SVC
from tensorflow.keras.optimizers import Adam



# In[2]:


def RF():
    model = RandomForestClassifier(n_estimators=100)
    return model

def SVM():
    model = SVC(C = 1000, gamma=0.001)
    return model

def CNN_base(dim, lr=0.0001):
    model = models.Sequential([
        Input([dim,dim,1]),
        Conv2D(16, 3, activation='relu', padding='same'),
        MaxPooling2D(pool_size=(2,2)),
        Conv2D(32, 3, activation='relu', padding='same'),
        MaxPooling2D(pool_size=(2,2)),
        Conv2D(64, 3, activation='relu', padding='same'),
        MaxPooling2D(pool_size=(2,2)),
        Conv2D(128, 3, activation='relu', padding='same'),
        MaxPooling2D(pool_size=(2,2)),
        Conv2D(256, 3, activation='relu', padding='same'),
        MaxPooling2D(pool_size=(2,2)),
        GlobalAveragePooling2D(),
        Dense(128, activation='relu'),
        Dense(128, activation='relu'),
        Dense(9, activation='softmax')
    ])
    model.compile(optimizer= Adam(lr=0.0001),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def FNN(dim):
    model = models.Sequential([
        Input([59,]),
        Dense(128, activation='relu'),
        Dense(128, activation='relu'),
        Dense(9, activation='softmax')
    ])
    model.compile(optimizer=Adam(lr=0.0001),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def JointNN(dim):
    
    input_feature_vector = Input((59,))
    
    input_wm_tensor = Input((dim, dim, 1))
    conv_1 = Conv2D(16, (3,3), activation='relu', padding='same')(input_wm_tensor)
    pool_1 = MaxPooling2D(pool_size=(2, 2),strides=2 ,padding='same')(conv_1)
    conv_2 = Conv2D(32, (3,3), activation='relu', padding='same')(pool_1)
    pool_2 = MaxPooling2D(pool_size=(2, 2),strides=2 ,padding='same')(conv_2)
    conv_3 = Conv2D(64, (3,3), activation='relu', padding='same')(pool_2)
    pool_3 = MaxPooling2D(pool_size=(2, 2),strides=2 ,padding='same')(conv_3)
    conv_4 = Conv2D(64, (3,3), activation='relu', padding='same')(pool_2)
    pool_4 = MaxPooling2D(pool_size=(2, 2),strides=2 ,padding='same')(conv_3)
    conv_5 = Conv2D(64, (3,3), activation='relu', padding='same')(pool_2)
    pool_5 = MaxPooling2D(pool_size=(2, 2),strides=2 ,padding='same')(conv_3)
    gap = GlobalAveragePooling2D()(pool_5)

    concat_vector = concatenate([input_feature_vector, gap])

    concat_hidden_1 = Dense(128, activation='relu')(concat_vector)
    concat_hidden_2 = Dense(128, activation='relu')(concat_hidden_1)
    prediction = Dense(9, activation='softmax')(concat_hidden_2)

    model = models.Model([input_feature_vector, input_wm_tensor], prediction)
    model.compile(optimizer=Adam(lr=0.0001),
                 loss='sparse_categorical_crossentropy',
                 metrics=['accuracy'])
    return model
    

# In[ ]:




