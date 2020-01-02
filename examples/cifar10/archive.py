
import os
import cv2
import pickle
import numpy as np

WIDTH = 32
HEIGHT=32
LENGTH=1024
DATA_BATCH_SIZE=5
    
def unpickle(file):
    with open(file, 'rb') as fo:
        #dict = pickle.load(fo, encoding='bytes')
        dict = pickle.load(fo)
    return dict

pwd = os.getcwd()

def release_data(filename, dst_dir):
    pack = unpickle(filename)
    labels = pack[b'labels']
    data = pack[b'data']
    filelist = pack[b'filenames']
    
    num = len(labels)

    for i in range(num):
        label = labels[i]
        im = data[i]
        file = filelist[i]
        
        R = np.array(im[:LENGTH]).reshape(HEIGHT, WIDTH)
        G = np.array(im[LENGTH:LENGTH*2]).reshape(HEIGHT, WIDTH)
        B = np.array(im[LENGTH*2:]).reshape(HEIGHT, WIDTH)
        
        img = cv2.merge([B, G, R])
        #cv2.imshow("1", img)
        #cv2.waitKey(0)

        with open(dst_dir + '.txt', "a+") as fal:
            fal.write(os.path.join(pwd, dst_dir, file.decode() + ' ' + str(label)) + '\n')
        
        if os.path.exists(dst_dir):
            cv2.imwrite(os.path.join(dst_dir, file.decode()), img)

            

if '__main__' == __name__:

    for i in range(DATA_BATCH_SIZE):
        filename = 'cifar-10-batches-py/data_batch_' + str(i+1)
        release_data(filename, 'train')
        print('%s release completed' % filename)
        
    print('train data completed')
    
    release_data('cifar-10-batches-py/test_batch', 'test')
    print('test data completed')
        