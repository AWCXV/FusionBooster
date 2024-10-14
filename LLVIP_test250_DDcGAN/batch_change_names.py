import os
import time
import shutil

def main():

    test_path = "./"
    output_path = './';
    # if mode == 'L':
        # image = imread(path, mode=mode)
    # elif mode == 'RGB':
        # image = Image.open(path).convert('RGB')
    num_imgs = 50;
    for i in range(1,1+num_imgs):
        print('processing '+str(i)+'-th image...');
        fileNameA = str(i)+'.png';
        fileNameB = str(200+i)+'.png';

        os.rename(fileNameA,fileNameB);

if __name__ == '__main__':
    main()
