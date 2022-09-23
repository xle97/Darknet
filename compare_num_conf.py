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
    '''
    该代码用于保存与另外一个结果按图片名，预测框类别尽可能一致的文件
    '''

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
    with open('/home/rzhang/Desktop/new_result/num_conf_txt/darknet_num_conf.txt', 'a') as f:

        for name ,box2 in dict2.items():
            f.write(name)
            if len(dict1[name][0]) == 0 :
                f.write('=>0\n[]\n')
            else:
                sim = 0
                cnt =0
                flag={}
                
                check = len(dict1[name])
                f.write('=>{}\n'.format(check))
                for j in dict1[name]:
                    flag[j] = 0
                for i in dict2[name]:   
                    maxbox = 'no'
                    index2=i[1:-1].split(' ')[0]
                    for j in dict1[name]:  
                        index1=j[1:-1].split(' ')[0]
                        if index1 == index2:
                            u= compute_IOU(j[1:-1].split(' ')[2:],i[1:-1].split(' ')[2:])
                            
                            maxbox = j if sim < u else maxbox   #求得最大一个匹配框
                            sim = max(u,sim)
                    if maxbox == 'no':
                        continue
                    else:
                        if flag[maxbox]==0:
                            ttt = maxbox[1:-1].split(' ')
                            flag[maxbox]=1
                            f.write("[{} {:.3f} {:.2f} {:.2f} {:.2f} {:.2f}] ".format(ttt[0],float(ttt[1]),float(ttt[2]),float(ttt[3]),float(ttt[4]),float(ttt[5])))
                            cnt +=1
                for j in dict1[name]:
                    if flag[j] == 0:
                        ttt = j[1:-1].split(' ')
                        flag[j] = 1
                        f.write("[{} {:.3f} {:.2f} {:.2f} {:.2f} {:.2f}] ".format(ttt[0],float(ttt[1]),float(ttt[2]),float(ttt[3]),float(ttt[4]),float(ttt[5])))
                        cnt+=1
                if cnt!= check:
                    print(name)
                f.write("\n")
   