import os
import glob

def get_label(fname):
    label = int(fname.split('.png')[0][-1])
    return label

file_root1 = './data/train/'
files = os.listdir(file_root1)
files.sort()
files_1 = [f for f in files if '.png' in f]

f= open("y_train.txt","w+")
i=1
for a in files_1:
	st=str(i)+" ./data/train/"+a+" "+str(get_label(a))
	f.write("%s \n"%st)
	i=i+1