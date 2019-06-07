# tb_classification

file structure:
  1) data_preprocess.ipynb: Python Notebook to preprocess image to increase the data size the images are agumented and then the test and train folder are created.
  2) data_test.py and data_train.py : are used to create pickle format of data with images and y vector
  3) y_vector.py: is used to create y vector for corresponding data by extracting labels from the image name.
  4) y_train.txt and y_test.txt : it contains the y vector of images in training and testing respectively.
  5) train.py : contains the model used for training 
  6) test.py : used to test on the trained model

