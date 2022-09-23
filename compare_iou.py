import os
from tabnanny import check

path2 = '/home/rzhang/Desktop/new_result/239_out.txt'
path1 = '/home/rzhang/Desktop/new_result/darknet_out.txt'


def str2float(l_str):   ##保留小数位，取整数位，测试最高的iou
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
    Miou=0
    count=0
    tmp = 0
    print("darknet漏检框:")
    for name ,box2 in dict2.items():
        iou =0
        if len(dict2[name][0]) == 0 and len(dict1[name][0]) == 0:
            iou=1
        elif len(dict2[name][0]) == 0 and len(dict1[name][0]) > 0:
            iou =0
        else:
            for i in dict2[name]:   #计算同一个name的iou
                ttt = i[1:-1].split(' ')
                index2=ttt[0]
                flag = 0
                sim = 0
                for j in dict1[name]:   ##求多个预测框匹配度最高的一个370.1302 89.974 13.5917 8.1243
                    index1=j[1:-1].split(' ')[0]
                    
                    if index1 == index2:
                        flag =1
                        u= compute_IOU(j[1:-1].split(' ')[2:],i[1:-1].split(' ')[2:])
                        sim = u if sim < u else sim
                if flag ==0:
                    print("{}=>{} {:.3f}".format(name,index2,float(ttt[1])))
                    tmp+=1

            
                iou+=sim
            iou /= len(dict2[name])
        # print("{} => {:.6f}".format(name,iou))
        Miou+=iou
        count+=1
    print("漏检框数量：{}".format(tmp))
    print("{:.6f}".format(Miou/count))




