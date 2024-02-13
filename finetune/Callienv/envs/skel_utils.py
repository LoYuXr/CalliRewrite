import os
import cv2
import numpy as np
import math
def EMA(old_skel, new_skel, gamma = 0.95):
    
    if old_skel.shape[0]-2 == new_skel.shape[0]:
        old_skel[1:-1] = gamma * old_skel[1:-1] + (1-gamma) * new_skel 
    else:
        old_skel[1:new_skel.shape[0]+1] = gamma * old_skel[1:new_skel.shape[0]+1]+ (1-gamma) * new_skel 

    return old_skel

def preprocess_imgs_strokes(img_base_dir, skel_base_dir):
    '''
    return a list of tuple, including the image dir and the corresponding skel_dir
    the img and stroke names should be the same.
    '''
    skel_list = [i for i in os.listdir(skel_base_dir)]
    return_list = []
    for i in os.listdir(img_base_dir):
        stroke_name = i.split('.')[0]+'.npz'
        if stroke_name not in skel_list:
            raise ValueError('image name and stroke name don\'t match.')
        return_list.append(tuple((img_base_dir+i, skel_base_dir+stroke_name)))
    
    return return_list

def transfer_data(skel_path):
    '''
    generate contour_image and transfer vectorized stroke
    8.14: may be npz or npy file
    '''
    if skel_path.split(".")[-1] == 'npz':
        # contour_img = extract_contour(img_path)
        skel = np.load(skel_path, encoding='latin1', allow_pickle=True)
        strokes_data = skel['strokes_data']
        init_cursors = skel['init_cursors']
        image_size = skel['image_size']
        round_length = skel['round_length']
        skel = make_global_nplist(strokes_data, init_cursors, image_size, round_length, cursor_type='next',
                                min_window_size=32, raster_size=128)
    else:
        skel = np.load(skel_path) 
        image_size = 256
    skel_list = parse_skel(skel, image_size)
    return skel_list

def extract_contour(dir):
    'removed in callienv4'
    img = cv2.imread(dir, 0)
    kernel = np.ones((2,2))
    img = cv2.dilate(img,kernel, 2)
    x = cv2.Sobel(img,cv2.CV_16S,1,0)
    y = cv2.Sobel(img,cv2.CV_16S,0,1)
    absX = cv2.convertScaleAbs(x)
    absY = cv2.convertScaleAbs(y)
    dst = cv2.addWeighted(absX,0.5,absY,0.5,0)
    ret, binary = cv2.threshold(dst,127,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    
    binary = cv2.dilate(binary,kernel, 2)
    return binary

def parse_skel(skel_list, image_size):
    '''
    origin: [p_t, x1, y1, x2, y2, x3, y3]
    transfer to: [0/1, xi, yi]  -> sample points from the vectorized
    '''
    cur_list = []
    flag = True
    

    for i in range(skel_list.shape[0]):
        if skel_list[i][0] == 1: # quit
            flag = False
        elif skel_list[i][0] == 0:
            x1y1, x2y2, x3y3 = skel_list[i][1:3], skel_list[i][3:5],skel_list[i][5:7]
            len = x3y3-x1y1
            len = int(math.sqrt(len[0]**2+len[1]**2))
            if len > 20 and len <= 40: len =  int(len/3)
            if len > 40: len = int(len/4)
            for i in range(len):
                t = i/len
                pos = ((1-t) * (1-t) * x1y1 + 2 * t * (1-t) * x2y2 + t * t * x3y3) / image_size  ## 归一到0-1
                if flag == False:
                    flag = True
                    cur_list.append([1,pos[0],pos[1]])
                else:
                    cur_list.append([0,pos[0],pos[1]])

            vec0 = cur_list[-1][1]-cur_list[-2][1]
            vec1 =  cur_list[-1][2]-cur_list[-2][2]
            cur_list.append([0, cur_list[-1][1] + 0.05*vec0, cur_list[-1][2] + 0.05*vec1])
            cur_list.append([0, cur_list[-1][1] + 0.05*vec0, cur_list[-1][2] + 0.05*vec1])
    
    return np.array(cur_list)

def add_beg_end_seq(skel_list, contours, width):
    skel_list = skel_list * width
    
    pt_1 = np.where(skel_list[:,0]==width)[0]
    pt_1 = np.append(np.insert(pt_1, 0, 0), len(skel_list))

    split_strokes = [skel_list[pt_1[j-1]:pt_1[j]] for j in range(1,len(pt_1))]
    
    for num in range(len(split_strokes)):
        id  = split_strokes[num]
        if len(id) <3:
            continue
        front, bottom = id[0], id[-1]
        front_list, bottom_list = [], []
        for con in contours:
            
            dis_front = abs(cv2.pointPolygonTest(con, tuple(front[1:]), True))
            dis_bottom = abs(cv2.pointPolygonTest(con, tuple(bottom[1:]), True))
            front_list.append(dis_front)
            bottom_list.append(dis_bottom)
        
        front_r = min(front_list)
        bottom_r = min(bottom_list)

        if front_r <= 1.5 or bottom_r <= 1.5:
            continue
        else:
            front_r, bottom_r = min(10, front_r), min(10, bottom_r)
            vec_front = id[0,1:] - id[2,1:]
            vec_bottom = id[-1,1:] - id[-3,1:]
            vec_front, vec_bottom = vec_front/np.linalg.norm(vec_front), vec_bottom / np.linalg.norm(vec_bottom)
            control_front = id[0,1:]+vec_front*front_r
            control_bottom = id[-1,1:]+vec_bottom*bottom_r
            start_front = id[0,1:]+np.array([-vec_front[1], vec_front[0]])* 0.3*front_r
            end_bottom = id[-1,1:]+np.array([-vec_bottom[1], vec_bottom[0]])* 0.3*bottom_r
            front_append, bottom_append = [],[]

            for i in range(8):
                t = i/8
                b = (i+1)/8
                front_pos = ((1-t) * (1-t) * start_front + 2 * t * (1-t) * control_front + t * t * front[1:])
                bottom_pos = ((1-b) * (1-b) * bottom[1:] + 2 * b * (1-b) * control_bottom + b * b * end_bottom)
                front_append.append(front_pos)
                bottom_append.append(bottom_pos)

            front_append, bottom_append = np.array(front_append), np.array(bottom_append)
            front_append = np.hstack((np.zeros(front_append.shape[0])[:, np.newaxis], front_append))
            bottom_append = np.hstack((np.zeros(bottom_append.shape[0])[:, np.newaxis], bottom_append))

            if front[0] == width:
                id[0][0] = 0.0
                front_append[0,0] = width

            split_strokes[num] = np.vstack((front_append, id, bottom_append)).astype(float)

    skel_list = np.vstack([split_strokes[i] for i in range(len(split_strokes))])
    skel_list = skel_list / width

    return skel_list

            
def make_global_nplist(data, init_cursor, image_size, infer_lengths, cursor_type='next', min_window_size=32, raster_size=128):
    """
    parse one image (data sequence)
    :param data: (N_strokes, 9): flag, x0, y0, x1, y1, x2, y2, r0, r2
    :return: 
    """
    cur_list = []
    cursor_idx = 0

    if init_cursor.ndim == 1:
        init_cursor = [init_cursor]

    for round_idx in range(1):
        round_length = infer_lengths[round_idx]
        ## cursor_pos!!!
        cursor_pos = init_cursor[cursor_idx]  # (2)
        cursor_idx += 1
        
        prev_scaling = 1.0
        prev_window_size = float(raster_size)  # (1)
        
        
        for round_inner_i in range(round_length):

            stroke_idx = np.sum(infer_lengths[:round_idx]).astype(np.int32) + round_inner_i

            curr_window_size_raw = prev_scaling * prev_window_size
            curr_window_size_raw = np.maximum(curr_window_size_raw, min_window_size)
            curr_window_size_raw = np.minimum(curr_window_size_raw, image_size)
            curr_window_size = int(round(curr_window_size_raw))

            pen_state = data[stroke_idx, 0]
            
            stroke_params = data[stroke_idx, 1:]

            x1y1, x2y2 = stroke_params[0:2], stroke_params[2:4]
            x0y0 = np.zeros_like(x2y2)
            x0y0 = np.divide(np.add(x0y0, 1.0), 2.0)
            x2y2 = np.divide(np.add(x2y2, 1.0), 2.0)
            x1y1[0] = x0y0[0] + (x2y2[0] - x0y0[0]) * x1y1[0]
            x1y1[1] = x0y0[1] + (x2y2[1] - x0y0[1]) * x1y1[1]

            padding_size = int(np.ceil(curr_window_size_raw / 2.0))
            cursor_pos1=cursor_pos*image_size

            x1y1_a = cursor_pos1 - curr_window_size_raw / 2.0
            x2y2_a = cursor_pos1 + curr_window_size_raw / 2.0

            x1y1_a_floor = np.floor(x1y1_a)
            x2y2_a_ceil = np.ceil(x2y2_a)

            cursor_pos_b_oricoord = (x1y1_a_floor + x2y2_a_ceil) / 2.0
            cursor_pos_b = (cursor_pos_b_oricoord - x1y1_a) / curr_window_size_raw * 128
            raster_size_b = (x2y2_a_ceil - x1y1_a_floor)
            image_size_b = raster_size
            window_size_b = raster_size * (raster_size_b / curr_window_size_raw)
            
            cursor_b_x, cursor_b_y = np.split(cursor_pos_b, 2, axis=-1)

            y1_b = cursor_b_y - (window_size_b[1] - 1.) / 2.
            x1_b = cursor_b_x - (window_size_b[0] - 1.) / 2.
            y2_b = y1_b + (window_size_b[1] - 1.)
            x2_b = x1_b + (window_size_b[0] - 1.)

            x_scale = 127/(x2_b - x1_b)
            y_scale = 127/(y2_b - y1_b)
            scale = np.array((x_scale, y_scale))
            o_point = np.array((x1_b[0] / 127, y1_b[0] / 127))

            x0y0g = (x0y0 - o_point) * scale[0]
            x1y1g = (x1y1 - o_point) * scale[0]
            x2y2g = (x2y2 - o_point) * scale[0]



            patch_o_point = cursor_pos*float(image_size)

            x0y0g = x0y0g*raster_size_b + [x1y1_a_floor[1],x1y1_a_floor[0]]
            x1y1g = x1y1g*raster_size_b +[x1y1_a_floor[1],x1y1_a_floor[0]]
            x2y2g = x2y2g*raster_size_b + [x1y1_a_floor[1],x1y1_a_floor[0]]


            pen_state = np.array([pen_state])

            global_point = np.concatenate([pen_state, x0y0g, x1y1g, x2y2g], axis=-1)
            cur_list.append(global_point)
            tmp = 1. / 100

            ## next scaling params
            next_scaling = stroke_params[5]
            prev_scaling = next_scaling
            prev_window_size = curr_window_size_raw

            ## update cursor point
            # update cursor_pos based on hps.cursor_type
            new_cursor_offsets = stroke_params[2:4] * (float(curr_window_size_raw) / 2.0)
            new_cursor_offset_next = new_cursor_offsets

            new_cursor_offset_next = np.concatenate([new_cursor_offset_next[1:2], new_cursor_offset_next[0:1]], axis=-1)

            cursor_pos_large = cursor_pos * float(image_size)
            stroke_position_next = cursor_pos_large + new_cursor_offset_next

            if cursor_type == 'next':
                cursor_pos_large = stroke_position_next
            else:
                raise Exception('Unknown cursor_type')

            cursor_pos_large = np.minimum(np.maximum(cursor_pos_large, 0.0), float(image_size - 1))
            cursor_pos = cursor_pos_large / float(image_size)
    
    return_array = np.array(cur_list)
   
    return return_array  #[p_t, x1, y1, x2, y2, x3, y3]: if p_t == 1: don't draw; else: draw bezier


def save_visualization(save_file, img_path, control_list, origin_list):
    img = cv2.imread(img_path)
    img1 = np.copy(img)

    save_file_2 = save_file.rstrip('.png')+'_gen.png'
    control_list = img.shape[0]*control_list
    control_list = np.round(control_list).astype(int)
    control_color = (0,255,0)

    for i in range (control_list.shape[0]-1):
        if control_list[i+1][0] != img1.shape[0]:
            cv2.line(img1, (control_list[i][2], control_list[i][1]), (control_list[i+1][2], control_list[i+1][1]),control_color, 1)
    
    cv2.imwrite(save_file_2, img1)
    
    if origin_list is not None:
        save_file_1 = save_file.rstrip('.png')+'_raw.png'
        origin_list = img.shape[0]*origin_list
        origin_list = np.round(origin_list).astype(int)
        origin_color = (0,0,255)
    
        for i in range (origin_list.shape[0]-1):
            if origin_list[i+1][0] != img.shape[0]:
                cv2.line(img, (origin_list[i][2], origin_list[i][1]), (origin_list[i+1][2], origin_list[i+1][1]),origin_color, 1)  
        cv2.imwrite(save_file_1, img)

