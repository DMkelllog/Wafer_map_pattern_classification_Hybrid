#!/bin/sh
#!/bin/bash

for rep in {0..1} 
do

  for train_size in {1..2} # train_size [500, 5000, 50000, 162946]
  do
  
    method=0 # method 0:MFE
    for MFE_model in {0..2} # model ['RF', 'FNN', 'SVM']
    do
    python run_model.py $rep $method $MFE_model 1 $train_size
    done
  
  
    method=1 # method 1:CNN
    CNN_dim=1 # dim 0:32, 1:64, 2:96, 3:128
    python run_model.py $rep $method 0 $CNN_dim $train_size
    
    method=2 # method 2:Joint
    python run_model.py $rep $method 0 $CNN_dim $train_size
    
    method=3 # method 3:SE
    python run_model.py $rep $method 0 $CNN_dim $train_size
    
  done

done