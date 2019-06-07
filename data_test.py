from PIL import Image
import numpy as np
import pickle

def Normalize(image,mean,std):
    for channel in range(3):
        image[:,:,channel]=(image[:,:,channel]-mean[channel])/std[channel]
    return image

data_test={}


with open("./y_test.txt") as f:
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
        data_test[int(id)]=image
 
data_test=np.array(list(data_test.values()))
f=open("./data_test","wb+")
pickle.dump(data_test,f,protocol=4)

test_y={}

with open("./y_test.txt") as f:
    lines=f.read().splitlines()
    for line in lines:
        id,y,a=line.split(" ",2)
        # print(id,y,a)
        #y=np.array([float(i) for i in box.split()],dtype=np.float32)
        test_y[int(id)]=a
        print(test_y[int(id)])
test_y=np.array(list(test_y.values()))
f=open("./test_y","wb+") 
pickle.dump(test_y,f,protocol=4)