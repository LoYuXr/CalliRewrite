import numpy as np

def process(data, func, alpha=0.04, beta=0.5): #array, [p_t, x, y, r]
    alpha = 0.04 #0.08 #size of image
    beta = 0.5 #1.2  # to control the width of the strokes
    record_x = []
    record_y = []
    record_z = []
    
    for i in range(data.shape[0]):
        
        p_t, x, y, r = data[i]  #单位是m
        
        x_, y_ = x*alpha, y*alpha
        r_ = r*alpha*beta 

        h = func(r_)
        h = h*100
        
        if h is None:
            if i  == 0:
                h = 0.1
            else:
                h = record_z[-1] * 100
        h -= 0.09  ## slightly modift
        h = h/100
        if p_t ==0:
            record_x.append(x_)
            record_y.append(y_)
            record_z.append(h)
        
        if p_t == 1:
            # 如果p_t是1， 那么寓意着新的序列开始，我们需要对首做操作：篆书方法or楷书方法首先实现
            if i == data.shape[0] - 1: #如果最后一个点是1，后续没了，我们不予操作
                continue
            
            if type ==1:##楷书，我们从左上角进入：
                record_x.append(x_ - 2 * r_)
                record_y.append(y_ - 2 * r_)
                record_z.append(0.02)  #离开纸面
                
            else:##如果隶书，起笔位置是原来靠上
                record_x.append(x_) 
                record_y.append(y_)
                record_z.append(0.02)
                
            record_x.append(x_) #加入当前点
            record_y.append(y_)
            record_z.append(h)
            
            if type == 0: #隶书，当前点为落笔点，沿着反向回锋，再下行
                nxt_vec = data[i + 1][1:3] - data[i][1:3]
                nxt_vec = nxt_vec / np.linalg.norm(nxt_vec)
                record_x.append(x_-2*r_ * nxt_vec[0])
                record_y.append(y_-2*r_ * nxt_vec[1])
                record_z.append(h)#回锋
                record_x.append(x_)
                record_y.append(y_)
                record_z.append(h)#下行
            
    for iter_ in range(5):

        record_x.append(record_x[-1])
        record_y.append(record_y[-1])
        record_z.append(record_z[-1]+0.015)


    return record_x, record_y, record_z