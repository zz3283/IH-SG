# -*- coding:utf-8 -*-

import os


class ImageRename():
    def __init__(self):

        self.path = '/mnt/data/ZJ/dataset/mydata/masks'
    def rename(self):
        filelist = os.listdir(self.path)
        print(filelist)
        total_num = len(filelist)
        for item in filelist:
            if item.endswith('.jpg'):
                src = os.path.join(os.path.abspath(self.path), item)
                line = item.split('.')
                print(line[0])
                word = line[0].split('_')
                dst = os.path.join(os.path.abspath(self.path), word[0]+'_'+word[1]+ '.jpg')
                print(src)
                print(dst)

                os.rename(src, dst)
                print('converting %s to %s ...' % (src, dst))



if __name__ == '__main__':
    newname = ImageRename()
    newname.rename()