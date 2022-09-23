import os
# from tabnanny import check

path2 = '/home/rzhang/Desktop/new_result/239_out.txt'
path1 = '/home/rzhang/Desktop/new_result/darknet_out.txt'


def str2float(l_str):
    new_l=[]
    for i in l_str:
        new_l.append(float(i))
    return new_l

def compute_IOU(rec1,rec2):   #左上角 w h
    rec1=str2float(rec1)
    rec2=str2float(rec2)
    left_column_max  = max(rec1[0],rec2[0])
    right_column_min = min(rec1[0]+rec1[2],rec2[0]+rec2[2])
    up_row_max       = max(rec1[1],rec2[1])
    down_row_min     = min(rec1[1]+rec1[3],rec2[1]+rec2[3])
    # 两矩形无相交区域的情况
    if left_column_max>=right_column_min or down_row_min<=up_row_max:
        return 0
    # 两矩形有相交区域的情况
    else:
        S1 = rec1[2]*rec1[3]    
        S2 = rec2[2]*rec2[3]
        S_cross = (down_row_min-up_row_max)*(right_column_min-left_column_max)
        return S_cross/(S1+S2-S_cross)


if __name__ == "__main__":

    l1=[]
    l2=[]
    with open(path1,'r') as f:
        for line in f:
            l1.append(line.strip())
    with open(path2,'r') as f:
        for line in f:
            l2.append(line.strip())
    dict1={}
    dict2={}
    for i in l2:
        for j in l1:
            name2,rec2 = i.split('=>')    
            name1,rec1 = j.split('=>')
            
            rec2=rec2[1:-2]
            rec1=rec1[1:-2]
            box2 = rec2.split(',')
            box1 = rec1.split(',')
            dict2[name2]=box2
            dict1[name1]=box1
    with open('/home/rzhang/Desktop/new_result/239_num_conf.txt', 'a') as f:

        for name ,box2 in dict2.items():
            f.write(name)
            if len(dict2[name][0]) == 0 :
                f.write('=>0\n[]\n')
            else:
                sim = 0
                cnt =0
                flag={}
                
                check = len(dict2[name])
                f.write('=>{}\n'.format(check))
                for i in dict2[name]:   
                    ttt = i[1:-1].split(' ')
                    f.write("[{} {} {:.2f} {:.2f} {:.2f} {:.2f}] ".format(ttt[0],float(ttt[1]),float(ttt[2]),float(ttt[3]),float(ttt[4]),float(ttt[5])))
                f.write("\n")
   