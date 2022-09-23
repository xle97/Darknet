###制作标签文件

from ast import Not
import os

path1 = '/home/rzhang/Desktop/new_result/239_out.txt'
path2 = '/home/rzhang/Desktop/new_result/darknet_out.txt'

'''predict
E.g.  image_name.txt

class1  confidence left top right bottom
class2

'''
outfile = None
with open (path1,'r') as f:
    for line in f:
        name,rec = line.strip().split('=>')
        if outfile is not None:
            outfile.close()
        outfile = open(os.path.join('/home/rzhang/Desktop/label_pre/pred',name+'.txt'),'w')
        if rec[1]==']':
            outfile.close()
        else:
            rec = rec[1:-2].split(',')
            for box in rec:
                cls,conf,x,y,w,h= box[1:-1].split(' ')
                outfile.write("{} {} {} {} {} {}\n".format(cls,conf,x,y,float(x)+float(w),float(y)+float(h)))
'''groundtruth
E.g. image_name.txt
class1 left top right bottom
class2

'''
if outfile is not None:
    outfile.close()
with open(path2, 'r') as f:
    for line in f:
        name,rec = line.strip().split('=>')
        if outfile is not None:
            outfile.close()
        outfile = open(os.path.join('/home/rzhang/Desktop/label_pre/label',name+'.txt'),'w')
        if rec[1]==']':
            outfile.close()
        else:
            rec = rec[1:-2].split(',')
            for box in rec:
                cls,conf,x,y,w,h= box[1:-1].split(' ')
                outfile.write("{} {} {} {} {}\n".format(cls,x,y,float(x)+float(w),float(y)+float(h)))