#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from skimage.transform import resize, radon
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from skimage import measure
from scipy import interpolate, stats
import time
import pickle


# In[6]:


def load_wm_data():
    return pd.read_pickle("./data/WM811K.pkl")


# In[7]:


def find_wm_dim(x):
    dim0=np.size(x,axis=0)
    dim1=np.size(x,axis=1)
    return dim0,dim1


# In[13]:


def preprocess_wm_df(df):
    df = df.drop(['waferIndex','trianTestLabel','lotName'], axis = 1)
    df['waferMapDim']=df.waferMap.apply(find_wm_dim)
    
    df['failureNum']=df.failureType
    mapping_type={'Center':0,'Donut':1,'Edge-Loc':2,'Edge-Ring':3,'Loc':4,'Random':5,'Scratch':6,'Near-full':7,'none':8}
    df=df.replace({'failureNum':mapping_type})

    df_withlabel = df[(df['failureNum']>=0) & (df['failureNum']<=8)]
    df_withlabel = df_withlabel.reset_index()
    df_withlabel = df_withlabel.drop(df_withlabel[df_withlabel['dieSize']<100].index.tolist()).reset_index()
    y = np.array(df_withlabel['failureNum'])
    y = np.asarray(y).astype(np.int)
    return df_withlabel, y


# In[14]:


def load_preprocessed_df():
    df = load_wm_data()
    df, y = preprocess_wm_df(df)
    return df, y


# In[15]:


def resize_wm(WaferMapList,dim, lite=False):
    X = WaferMapList
    X_binary = [np.where(x<=1,0,1) for x in X]
    X_resize = [resize(x,(dim,dim), preserve_range=True, anti_aliasing=False) for x in X_binary]
    X_resize = np.array(X_resize).reshape(-1,dim,dim,1)
    if lite is True:
        X_resize = X_resize.astype(np.float16)
    return X_resize


# In[29]:


def export_resized_wafer(dim, lite=False):
    df, y = load_wm_data()
    X_resize = resize_wm(df.waferMap, dim, lite)
    if lite is True:
        filename = './data/X_'+str(dim)+'_lite.pickle'
    else:
        filename = './data/X_'+str(dim)+'.pickle'
    with open(filename, 'wb') as f:
        pickle.dump(X_resize, f, protocol=4)


# In[30]:


# **************** Density **********************
def cal_den(x):
    return 100*(np.sum(x==2)/np.size(x))  

def find_regions(x):
    rows=np.size(x,axis=0)
    cols=np.size(x,axis=1)
    if cols//5 == 0 or rows//5 == 0:
        fea_reg_den = [[0]]*13
        print("Warning! Zero division row:{} col:{}" .format(rows, cols))
        return fea_reg_den
    ind1=np.arange(0,rows,rows//5)
    ind2=np.arange(0,cols,cols//5)
    
    
    reg1=x[ind1[0]:ind1[1],:]
    reg3=x[ind1[4]:,:]
    reg4=x[:,ind2[0]:ind2[1]]
    reg2=x[:,ind2[4]:]

    reg5=x[ind1[1]:ind1[2],ind2[1]:ind2[2]]
    reg6=x[ind1[1]:ind1[2],ind2[2]:ind2[3]]
    reg7=x[ind1[1]:ind1[2],ind2[3]:ind2[4]]
    reg8=x[ind1[2]:ind1[3],ind2[1]:ind2[2]]
    reg9=x[ind1[2]:ind1[3],ind2[2]:ind2[3]]
    reg10=x[ind1[2]:ind1[3],ind2[3]:ind2[4]]
    reg11=x[ind1[3]:ind1[4],ind2[1]:ind2[2]]
    reg12=x[ind1[3]:ind1[4],ind2[2]:ind2[3]]
    reg13=x[ind1[3]:ind1[4],ind2[3]:ind2[4]]
    
    fea_reg_den = []
    fea_reg_den = [cal_den(reg1),cal_den(reg2),cal_den(reg3),cal_den(reg4),cal_den(reg5),cal_den(reg6),cal_den(reg7),cal_den(reg8),cal_den(reg9),cal_den(reg10),cal_den(reg11),cal_den(reg12),cal_den(reg13)]
    return fea_reg_den

# ****************** Radon ********************

def change_val(img):
    img[img==1] =0  
    return img


def cubic_inter_mean(img):
    theta = np.linspace(0., 180., max(img.shape), endpoint=False)
    sinogram = radon(img, theta=theta)
    xMean_Row = np.mean(sinogram, axis = 1)
    x = np.linspace(1, xMean_Row.size, xMean_Row.size)
    y = xMean_Row
    f = interpolate.interp1d(x, y, kind = 'cubic')
    xnew = np.linspace(1, xMean_Row.size, 20)
    ynew = f(xnew)/100   # use interpolation function returned by `interp1d`
    return ynew

def cubic_inter_std(img):
    theta = np.linspace(0., 180., max(img.shape), endpoint=False)
    sinogram = radon(img, theta=theta)
    xStd_Row = np.std(sinogram, axis=1)
    x = np.linspace(1, xStd_Row.size, xStd_Row.size)
    y = xStd_Row
    f = interpolate.interp1d(x, y, kind = 'cubic')
    xnew = np.linspace(1, xStd_Row.size, 20)
    ynew = f(xnew)/100   # use interpolation function returned by `interp1d`
    return ynew 

# ****************** Geometry ********************

def cal_dist(img,x,y):
    dim0=np.size(img,axis=0)    
    dim1=np.size(img,axis=1)
    dist = np.sqrt((x-dim0/2)**2+(y-dim1/2)**2)
    return dist  

def fea_geom(img):
    norm_area=img.shape[0]*img.shape[1]
    norm_perimeter=np.sqrt((img.shape[0])**2+(img.shape[1])**2)
    
    img_labels = measure.label(img, neighbors=4, connectivity=1, background=0)

    if img_labels.max()==0:
        img_labels[img_labels==0]=1
        no_region = 0
    else:
        info_region = stats.mode(img_labels[img_labels>0], axis = None)
        no_region = info_region[0][0]-1       
    
    prop = measure.regionprops(img_labels)
    prop_area = prop[no_region].area/norm_area
    prop_perimeter = prop[no_region].perimeter/norm_perimeter 
    
    prop_cent = prop[no_region].local_centroid 
    prop_cent = cal_dist(img,prop_cent[0],prop_cent[1])
    
    prop_majaxis = prop[no_region].major_axis_length/norm_perimeter 
    prop_minaxis = prop[no_region].minor_axis_length/norm_perimeter  
    prop_ecc = prop[no_region].eccentricity  
    prop_solidity = prop[no_region].solidity  
    
    return prop_area,prop_perimeter,prop_majaxis,prop_minaxis,prop_ecc,prop_solidity


# In[31]:


def extract_features(df):
    start1 = time.time()
    df['fea_reg']=df.waferMap.apply(find_regions)
    print('density ',(time.time()-start1)/60,'min')
    start = time.time()
    df['new_waferMap']=df.waferMap.apply(change_val)
    df['fea_cub_mean'] =df.waferMap.apply(cubic_inter_mean)
    df['fea_cub_std'] =df.waferMap.apply(cubic_inter_std)
    print('radon ',(time.time()-start)/60,'min')
    
    start = time.time()
    df['fea_geom'] =df.waferMap.apply(fea_geom)
    print('geometry ?',(time.time()-start)/60,'min')
    
    df_all=df.copy()

    a=[df_all.fea_reg[i] for i in range(df_all.shape[0])] #13
    b=[df_all.fea_cub_mean[i] for i in range(df_all.shape[0])] #20
    c=[df_all.fea_cub_std[i] for i in range(df_all.shape[0])] #20
    d=[df_all.fea_geom[i] for i in range(df_all.shape[0])] #6
    fea_all = np.concatenate((np.array(a),np.array(b),np.array(c),np.array(d)),axis=1) #59 in total
    print('total',(time.time()-start1)/60,'min')
    return fea_all


# In[32]:


def export_extracted_features():
    df, y = load()
    X_fe = extract_features(df)
    with open('./data/X_fe.pickle', 'wb') as f:
        pickle.dump(X_FE, f)


# In[33]:


def export_label():
    df, y = load()
    with open('./data/y.pickle', 'wb') as f:
        pickle.dump(y, f)


# In[34]:


def import_wm_data(mode, dim=32):
    
    if mode == 'CNN':
        with open('./data/X_'+str(dim)+'_lite.pickle', 'rb') as f:
            X_resize = pickle.load(f)
        with open('./data/y.pickle', 'rb') as f:
            y = pickle.load(f)
        return X_resize, y
    
    elif mode == 'MFE':
        with open('./data/X_fe.pickle', 'rb') as f:
            X_fe = pickle.load(f)
        with open('./data/y.pickle', 'rb') as f:
            y = pickle.load(f)
        return X_fe, y
    
    elif mode in ['SE', 'Joint']: # SE or JointNN
        with open('./data/X_'+str(dim)+'_lite.pickle', 'rb') as f:
            X_resize = pickle.load(f)
        with open('./data/X_fe.pickle', 'rb') as f:
            X_fe = pickle.load(f)
        with open('./data/y.pickle', 'rb') as f:
            y = pickle.load(f)
        return [X_fe, X_resize], y
    else:
        print('Invalid method')


# In[35]:


def split_data(X, y, RAND_NUM=777):
    X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, test_size=10000, 
                                                              random_state=RAND_NUM)
    X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval, test_size=0.2, 
                                                              random_state=RAND_NUM)
    X_split = [X_train, X_val, X_test]
    y_split = [y_train, y_val, y_test]
    
    return X_split, y_split


# In[36]:


def standard_scale(X_split):
    scaler = StandardScaler()
    X_train, X_val, X_test = X_split
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    return [X_train, X_val, X_test]


# In[ ]:




