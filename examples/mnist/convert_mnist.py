
from PIL import Image
import struct


def read_image(filename, dst_dir):
    f = open(filename, 'rb')
    buf = f.read()
    f.close()

    index = 0
    magic, num, rows, columns = struct.unpack_from('>IIII' , buf , index)
    index += struct.calcsize('>IIII')

    for i in xrange(num):
        image = Image.new('L', (columns, rows))

        for x in xrange(rows):
            for y in xrange(columns):
                image.putpixel((y, x), int(struct.unpack_from('>B', buf, index)[0]))
                index += struct.calcsize('>B')

        image.save(dst_dir + '/' + str(i) + '.png')


def read_label(filename, saveFilename):
    f = open(filename, 'rb')
    index = 0
    buf = f.read()

    f.close()

    magic, labels = struct.unpack_from('>II' , buf , index)
    index += struct.calcsize('>II')
  
    labelArr = [0] * labels

    for x in xrange(labels):
        labelArr[x] = int(struct.unpack_from('>B', buf, index)[0])
        index += struct.calcsize('>B')

    save = open(saveFilename, 'w')


    save.write('\n'.join(map(lambda x: str(x), labelArr)))
    save.write('\n')


    save.close()
    print 'save %s done!' % saveFilename


if __name__ == '__main__':
    read_image('train-images.idx3-ubyte', 'train')
    read_label('train-labels.idx1-ubyte', 'labels_train.txt')
    
    read_image('t10k-images.idx3-ubyte', 'test')
    read_label('t10k-labels.idx1-ubyte', 'labels_test.txt')




