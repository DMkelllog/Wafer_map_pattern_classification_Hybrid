#!/usr/bin/env python
# coding: utf-8

# In[1]:


from preprocess import import_wm_data, split_data, standard_scale
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from model import CNN_base, FNN, RF, SVM, JointNN, Ridge
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import f1_score, confusion_matrix
import pickle


# In[ ]:


class Classifier():
    
    def __init__(self):
        self.es = EarlyStopping(monitor='val_loss',patience=10, mode='auto', restore_best_weights=True)
        self.BATCH_SIZE = 256 * 4
        self.MAX_EPOCH = 1000
    
    def MFE(self, X_split, y_split, model):
        if model == 'SVM':
            X_split_scaled = standard_scale(X_split)
            Model = SVM()
            Model.fit(X_split_scaled[0], y_split[0])
            y_hat = Model.predict(X_split_scaled[2])
        
        elif model == 'RF':
            Model = RF()
            Model.fit(np.concatenate([X_split[0],X_split[1]]), np.concatenate([y_split[0],y_split[1]]))
            y_hat = Model.predict(X_split[2])
        
        elif model == 'FNN':
            X_split_scaled = standard_scale(X_split)
            Model = FNN(model)
            Model.fit(X_split_scaled[0], y_split[0],
                    validation_data=[X_split_scaled[1], y_split[1]],
                    epochs=self.MAX_EPOCH,
                    batch_size=self.BATCH_SIZE,
                    callbacks=[self.es], verbose=0)
            y_hat_prob_trainval = Model.predict(np.concatenate([X_split_scaled[0],X_split_scaled[1]]))
            y_hat_prob_test = Model.predict(X_split_scaled[2])
            y_hat = np.argmax(y_hat_prob_test, axis=1)
            return self.evaluate(y_split[2], y_hat), [y_hat_prob_trainval, y_hat_prob_test]
            
        else:
            print('model undefined')
        
        return self.evaluate(y_split[2], y_hat), 0
    
    def CNN(self, X_split, y_split, dim):
        Model = CNN_base(dim)
        Model.fit(X_split[0], y_split[0],
                    validation_data=[X_split[1], y_split[1]],
                    epochs=self.MAX_EPOCH,
                    batch_size=self.BATCH_SIZE,
                    callbacks=[self.es], verbose=0)
        y_hat_prob_trainval = Model.predict(np.concatenate([X_split[0],X_split[1]]))
        y_hat_prob_test = Model.predict(X_split[2])
        y_hat = np.argmax(y_hat_prob_test, axis=1)
        return self.evaluate(y_split[2], y_hat), [y_hat_prob_trainval, y_hat_prob_test]
        
    def Joint(self, X_split_mfe, X_split_cnn, y_split, dim):
        Model = JointNN(dim)
        Model.fit([X_split_mfe[0],X_split_cnn[0]], y_split[0],
                    validation_data=[[X_split_mfe[1],X_split_cnn[1]], y_split[1]],
                    epochs=self.MAX_EPOCH,
                    batch_size=self.BATCH_SIZE,
                    callbacks=[self.es], verbose=0)
        y_hat_prob = Model.predict([X_split_mfe[2],X_split_cnn[2]])
        y_hat = np.argmax(y_hat_prob, axis=1)
        return self.evaluate(y_split[2], y_hat), 0
    
    def get_prob(self, rep_id, training_size):
        # MFE prob
        filename_MFE = './result/{0}_{1}_'.format(rep_id, training_size)+'MFE_FNN'
        with open(filename_MFE+'_prob_train.pickle', 'rb') as f:
            prob_mfe_train = pickle.load(f)
        with open(filename_MFE+'_prob_test.pickle', 'rb') as f:
            prob_mfe_test = pickle.load(f)
        # CNN prob
        filename_MFE = './result/{0}_{1}_'.format(rep_id, training_size)+'CNN_64'
        with open(filename_MFE+'_prob_train.pickle', 'rb') as f:
            prob_cnn_train = pickle.load(f)
        with open(filename_MFE+'_prob_test.pickle', 'rb') as f:
            prob_cnn_test = pickle.load(f)
        return prob_mfe_train, prob_cnn_train, prob_mfe_test, prob_cnn_test
    
    def SE(self, y_split, rep_id, training_size=0):
        prob_mfe_train, prob_cnn_train, prob_mfe_test, prob_cnn_test = self.get_prob(rep_id, training_size)

        Model = Ridge(alpha=0.1)
        y_target = np.eye(9)[np.concatenate([y_split[0], y_split[1]])]
        Model.fit(np.concatenate([prob_mfe_train, prob_cnn_train], axis=1), y_target)
        y_hat_prob = Model.predict(np.concatenate([prob_mfe_test, prob_cnn_test], axis=1))
        y_hat = np.argmax(y_hat_prob, axis=1)
        return self.evaluate(y_split[2], y_hat), 0
    
    
    def evaluate(self, y_true, y_hat):
        macro = f1_score(y_true, y_hat, average='macro')
        micro = f1_score(y_true, y_hat, average='micro')
        cm = (confusion_matrix(y_true, y_hat))
        print('\n\nmacro: {}, micro: {}\n\n' .format(macro, micro))
        return macro, micro, cm
        
    def sample_training_size(self, X, y, training_size, rep_id):
        if training_size > 0 and training_size < 162946:
            X_new,_, y_new,_ = train_test_split(X, y, train_size=training_size+10000, random_state = 777+rep_id)
        else:
            X_new, y_new = X, y
        return X_new, y_new
    
    def classifier(self, rep_id, method, model, training_size=0):
        print('\n\n\nrep:     {}\nmethod:  {}\nmodel:   {}\n\n' .format(str(rep_id+1), method, model))
        # data import
        X, y = import_wm_data(mode=method, dim=model)
        
        if len(X) == 2:
            X_split_mfe, X_split_cnn, y_split = self.split_data_multi_model(X, y, training_size, rep_id)
        else:
            X_split, y_split = self.split_data_single_model(X, y, training_size, rep_id)
        
        # build each model 
        if method == 'MFE':
            return self.MFE(X_split, y_split, model)
        elif method == 'CNN':
            return self.CNN(X_split, y_split, model)
        elif method == 'Joint':
            return self.Joint(X_split_mfe, X_split_cnn, y_split, model)
        elif method == 'SE':
            return self.SE(y_split, rep_id, training_size)
    #####################################################
    
    def split_data_multi_model(self, X, y, training_size, rep_id):
        X_mfe, X_cnn = X[0], X[1]
        X_mfe_sampled, y_sampled = self.sample_training_size(X_mfe, y, training_size, rep_id)
        X_split_mfe, y_split = split_data(X_mfe_sampled, y_sampled, RAND_NUM=rep_id)
        
        X_cnn_sampled, _ = self.sample_training_size(X_cnn, y, training_size, rep_id)
        X_split_cnn, _ = split_data(X_cnn_sampled, y_sampled, RAND_NUM=rep_id)
        return X_split_mfe, X_split_cnn, y_split
    
    def split_data_single_model(self, X, y, training_size, rep_id):
        X_sampled, y_sampled = self.sample_training_size(X, y, training_size, rep_id)
        X_split, y_split = split_data(X_sampled, y_sampled, RAND_NUM=rep_id)
        return X_split, y_split

