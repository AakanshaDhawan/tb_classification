from PIL import Image
import numpy as np
import pickle

def Normalize(image,mean,std):
    for channel in range(3):
        image[:,:,channel]=(image[:,:,channel]-mean[channel])/std[channel]
    return image

data_train={}


with open("./y_train.txt") as f:
    lines=f.read().splitlines()
    for line in lines:
        id,path=line.split(" ",1)
        path=path.split(" ",1)        
        image=Image.open(path[0]).convert('RGB')
        image=image.resize((128,128))
        image=np.array(image,dtype=np.float32)
        image=image/255
        image=Normalize(image,[0.485,0.456,0.406],[0.229,0.224,0.225])
        print(image)
        data_train[int(id)]=image
 
data_train=np.array(list(data_train.values()))
f=open("./data_train","wb+")
pickle.dump(data_train,f,protocol=4)

train_y={}

with open("./y_train.txt") as f:
    lines=f.read().splitlines()
    for line in lines:
        id,y,a=line.split(" ",2)
        # print(id,y,a)
        #y=np.array([float(i) for i in box.split()],dtype=np.float32)
        train_y[int(id)]=a
        print(train_y[int(id)])
train_y=np.array(list(train_y.values()))
f=open("./train_y","wb+") 
pickle.dump(train_y,f,protocol=4)