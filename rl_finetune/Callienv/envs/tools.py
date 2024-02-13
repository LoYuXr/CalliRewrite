import numpy as np
import math
import cv2
from skimage.draw import polygon, disk, ellipse
from gym.error import DependencyNotInstalled
try:
    import pygame
    from pygame import gfxdraw
except ImportError:
    raise DependencyNotInstalled(
        "pygame is not installed, run `pip install gym[classic_control]`"
    )

class Tool_Base():
    '''
    This is a class to define the tool we use, the geometric properties the tool enjoys, 
    the plotting functions on the canvas and the downsampled_canvas, and the dynamics
    '''
    def __init__(self, r_min, r_max, l_min, l_max, theta_min, theta_max, theta_step):
  
        self.r_min = r_min
        self.r_max = r_max
        self.l_min = l_min
        self.l_max = l_max
        self.theta_min = theta_min  #0 degree
        self.theta_max = theta_max  #360 degrees
        self.theta_step = theta_step

    
    def calc_four_points(self, v_1, cur_r, cur_l, rad):
        raise NotImplementedError
     
    def geometric_r_l(self, contours, center, canvas_width):
        raise NotImplementedError
    
    def draw_canvas(self, canvas_width, four_points, current_r):
        raise NotImplementedError
   
    def dynamics(self, state, action, center, next_vec_x, next_vec_y, contours, canvas_width):
        raise NotImplementedError
    
    def reset(self, deg, center, canvas_width, contours):
        raise NotImplementedError
    
    def visualize_tool(self, display_point_arr, radii):
        raise NotImplementedError
    
    
class Writing_Brush(Tool_Base):
    def __init__(self, r_min, r_max, l_min, l_max, theta_min, theta_max, theta_step):
        super().__init__( r_min, r_max, l_min, l_max, theta_min, theta_max, theta_step)
        self.action_space = 2
        
    def calc_four_points(self, v_1, cur_r, cur_l, rad):
        ## tip point
        vec_tip = np.array([-math.sin(rad), math.cos(rad)])
        v_2 = v_1 + cur_l * vec_tip

        ##triangle points
        phi = math.acos(min(1, max(cur_r/cur_l, -1)))
        theta_1 = (rad - phi)%(2*math.pi)
        theta_2 = (rad + phi)%(2*math.pi)

        vec_th1 = np.array([-math.sin(theta_1),math.cos(theta_1)])
        vec_th2 = np.array([-math.sin(theta_2),math.cos(theta_2)])
        pt_th1 = v_1 + cur_r * vec_th1   ##right point
        pt_th2 = v_1 + cur_r * vec_th2   ##left point

        return np.array([v_1,pt_th1,v_2,pt_th2]), cur_r
    
    def geometric_r_l(self, contours, center, canvas_width):
        """
        For writing brush 
        Use cv2.findContours() to generate contour lists, and for each contour in contours,
        use cv2.pointPolygonTest(countour, (x,y), True) to return real minimal distance r.
        """
        center = (int(center[1]), int(center[0]))  # swap x and y coordinates to match cv2 coordinate system
        dis_list = []
        for con in contours:
            dis = abs(cv2.pointPolygonTest(con, center, True))
            dis_list.append(dis)
        next_r = min(dis_list) / canvas_width
        if next_r == 0: next_r = 5e-5

        next_r = min(next_r, 0.9*self.r_max)
        next_l = 0.003 + 2.021 * next_r  
        return next_r, next_l
    
    def draw_canvas(self, canvas_width, four_points, current_r):
        center = (canvas_width*four_points[0]).astype(int)
        center = np.clip(center, a_min=0, a_max=canvas_width-1)
        rr1, cc1 = disk(center, current_r)
        
        poly_pts = (canvas_width*four_points[1:]).astype(int)
        poly_pts = np.clip(poly_pts, a_min=0, a_max=canvas_width-1)
        rr2, cc2 = polygon(poly_pts[:, 0], poly_pts[:, 1])

        rr = np.concatenate((rr1, rr2))
        cc = np.concatenate((cc1, cc2))

        rr = np.clip(rr, a_min=0, a_max=canvas_width-1)
        cc = np.clip(cc, a_min=0, a_max=canvas_width-1)

        return rr,cc
    
    def brush_dynamics(self, angle, radius, length, angle_vec, skel_vec, angle_difference):
        ''' angle dynamics for virtual brush
            angle difference: differences between skelecton angle and previous angle.
        '''
        sign1 = np.cross(angle_vec, skel_vec)
        sign1 = sign1/abs(sign1)
        cos_sim = np.inner(angle_vec,skel_vec)/(np.linalg.norm(angle_vec)* np.linalg.norm(skel_vec))
        if cos_sim >=1:
            cos_sim = 1-1e-4
        if cos_sim<1:
            cos_sim = -1+1e-4
        sim = math.sqrt(1-cos_sim**2) #sin_sim

        ## modify in april: 3 step: [1, 0.707]: phase 1; [0.707, -0.966] phase 2; [-0.966, -1] phase 3
        if cos_sim <= 1 and cos_sim >= 0.707:
            delta_theta = self.theta_step * sim
        elif cos_sim < 0.707 and cos_sim > -0.96:
            delta_theta = self.theta_step * sim * (1+cos_sim) * math.sqrt(0.5/(radius+1e-4)) #会散开
        elif cos_sim <= -0.96 and cos_sim >= -1:  # turn!
            
            r_p = -cos_sim * radius
            delta_theta = 165
            l_p = 0.003 + 2.021 * r_p
            
            assert l_p >=0 and r_p >=0
            
            delta_theta = min(delta_theta, sign1*angle_difference)
            delta_theta = sign1 * delta_theta
            next_theta = (angle*self.theta_max + delta_theta)%self.theta_max

            return next_theta, {r_p, l_p}
        else:
            raise ValueError
            
        delta_theta = min(delta_theta, sign1*angle_difference)
        delta_theta = sign1 * delta_theta
        next_theta = (angle*self.theta_max + delta_theta)%self.theta_max

        return next_theta, {}
    
    def dynamics(self, state, action, center, next_vec_x, next_vec_y, contours, canvas_width):
        ## calculate r, l
        next_r, next_l = self.geometric_r_l(contours, center, canvas_width)
        ## preparation for theta calculation
        angle_vec = np.array([-math.sin(math.radians(state[3]*self.theta_max)),\
                               math.cos(math.radians(state[3]*self.theta_max))])
        skel_vec = np.array([next_vec_x, next_vec_y])
        if next_vec_x <=0:
            skel_angle = math.degrees(math.acos(next_vec_y))
        else:
            skel_angle = self.theta_max - math.degrees(math.acos(next_vec_y)) ##angle of the skelecton

        angle_diff = skel_angle-state[3]*self.theta_max
        ## theta dynamics
        next_theta, ret_set = self.brush_dynamics(state[3], next_r, next_l, angle_vec, skel_vec, angle_diff)
        # if len(ret_set) !=0:
        #     next_r, next_l = ret_set
        return next_r, next_l, next_theta
    
    def reset(self, deg, center, canvas_width, contours):
        re_theta = deg/self.theta_max
        re_r, re_l = self.geometric_r_l(contours, center, canvas_width)
        return re_r, re_l, re_theta
        
    
    def visualize_tool(self, screen, display_point_arr, radii):
        gfxdraw.aapolygon(
            screen,
            [(display_point_arr[3][1], display_point_arr[3][0]), 
            (display_point_arr[2][1], display_point_arr[2][0]), 
            (display_point_arr[1][1], display_point_arr[1][0])], 
            (0,0,70),
        )

        gfxdraw.filled_polygon(
            screen,
            [(display_point_arr[3][1], display_point_arr[3][0]), 
            (display_point_arr[2][1], display_point_arr[2][0]), 
            (display_point_arr[1][1], display_point_arr[1][0])], 
            (239,70,70),
        )
        
        ## draw circle
        gfxdraw.aacircle(
            screen,
            display_point_arr[0][1],
            display_point_arr[0][0],
            radii,
            (0,0,70),
        )
        gfxdraw.filled_circle(
            screen,
            display_point_arr[0][1],
            display_point_arr[0][0],
            radii,
            (239,70,70),
        )
        return screen


class Ellipse(Tool_Base):
    def __init__(self, r_min, r_max, l_min, l_max, theta_min, theta_max, theta_step):
        super().__init__( r_min, r_max, l_min, l_max, theta_min, theta_max, theta_step)
        self.action_space = 3

    def calc_four_points(self, v_1, cur_r, cur_l, rad):
        
        pt_th1 = np.zeros_like(v_1)
        pt_th1[0] = cur_r # semi-minor axis
        pt_th2 = np.zeros_like(v_1)
        pt_th2[0] = cur_l # semi-major axis
        v_2 = np.zeros_like(v_1)
        if rad<=math.pi:
            v_2[0] = rad
        else:
            v_2[0] = rad-2*math.pi
        return np.array([v_1,pt_th1,v_2,pt_th2]), cur_r

    def geometric_r_l(self, contours, center, canvas_width):
        center = (int(center[1]), int(center[0]))  # swap x and y coordinates to match cv2 coordinate system
        dis_list = []
        for con in contours:
            dis = abs(cv2.pointPolygonTest(con, center, True))
            dis_list.append(dis)

        next_r = min(dis_list) / canvas_width
        next_r = min(next_r, 1.5* self.r_max)
        next_l = 1.2*next_r    #tobemod
        return next_r, next_l
    
    def draw_canvas(self, canvas_width, four_points, current_r):
        center = (canvas_width*four_points[0]).astype(int)
        center = np.clip(center, a_min=0, a_max=canvas_width-1)

        l_r_list = (canvas_width*four_points[1:]).astype(int) # minor, angle, amjor
        # print(center[0], center[1], l_r_list[0][0], l_r_list[-1][0], four_points[2][0])
        rr1, cc1 = ellipse(center[0], center[1], l_r_list[0][0]+1e-4, l_r_list[-1][0]+1e-4, rotation=four_points[2][0])
        
        rr1 = np.clip(rr1, a_min=0, a_max=canvas_width-1)
        cc1 = np.clip(cc1, a_min=0, a_max=canvas_width-1)

        return rr1, cc1

    def dynamics(self, state, action, center, next_vec_x, next_vec_y, contours, canvas_width):
        #next_r, next_l, = state[1], state[2]
        next_r, next_l = self.geometric_r_l(contours, center, canvas_width)
        delta_theta = action[2]*self.theta_step  #degree
        next_theta =  (state[3]*self.theta_max+ delta_theta)%self.theta_max
        return next_r, next_l, next_theta
    
    def reset(self, deg, center, canvas_width, contours):
        re_theta = deg/self.theta_max
        re_r, re_l = self.geometric_r_l(contours, center, canvas_width)
        return re_r, re_l, re_theta
    
    def visualize_tool(self, screen, display_point_arr, radii):
        #size = screen.get_size()
        #ellipse_surface = pygame.Surface((size[0], size[1]), pygame.SRCALPHA)
        gfxdraw.aaellipse(
            screen,
            display_point_arr[0][1],
            display_point_arr[0][0],
            display_point_arr[3][0], #horizontal: semi-major
            display_point_arr[1][0], # semi-minor
            (0,0,70),  ## horizontal and vertical,没旋转是大问题！！
        )
        gfxdraw.filled_ellipse(
            screen,
            display_point_arr[0][1],
            display_point_arr[0][0],
            display_point_arr[3][0],
            display_point_arr[1][0],
            (239,70,70),  ## horizontal and vertical,没旋转是大问题！！
        )
        # rotated= pygame.transform.rotate(ellipse_surface, math.degrees(-display_point_arr[2][0])) #counterclockwise
        # size1 = rotated.get_size()
        # screen.blit(rotated, ((size[0] - size1[0])//2,(size[1] - size1[1])//2 ))
        # #rotated_surface = pygame.transform.rotate(ellipse_surface, math.degrees(display_point_arr[2][0]))
        
        return screen



class Chisel_Tip_Marker(Tool_Base):
    def __init__(self, r_min, r_max, l_min, l_max, theta_min, theta_max, theta_step):
        super().__init__( r_min, r_max, l_min, l_max, theta_min, theta_max, theta_step)
        self.action_space = 3

    def calc_four_points(self, v_1, cur_r, cur_l, rad):
        center = v_1
        #  变形矩阵 matrix mult 来计算, counter-clockwise

        init_pts = np.array([[cur_l/2, cur_l/2, -cur_l/2, -cur_l/2], [-cur_r/2, cur_r/2, cur_r/2, -cur_r/2]])
        rot_matrix = np.array([[math.cos(rad), -math.sin(rad)], [math.sin(rad), math.cos(rad)]])
        pts = np.dot(rot_matrix, init_pts).T

        pts = center + pts

        return pts, cur_r

    def geometric_r_l(self, contours, center, canvas_width):
        r = 0.039*2.2 ##TOBEMOD
        l = 0.018*1.1 ##TOBEMOD
        return r, l
    
    def draw_canvas(self, canvas_width, four_points, current_r):
        poly_pts = (canvas_width*four_points).astype(int)
        poly_pts = np.clip(poly_pts, a_min=0, a_max=canvas_width-1)
        rr2, cc2 = polygon(poly_pts[:, 0], poly_pts[:, 1])
        return rr2, cc2
    
    def dynamics(self, state, action, center, next_vec_x, next_vec_y, contours, canvas_width):
        next_r, next_l, = state[1], state[2]
        delta_theta = action[2]*self.theta_step  #action includes changing angle! theta
        next_theta =  (state[3]*self.theta_max+ delta_theta)%self.theta_max
        return next_r, next_l, next_theta
    
    def reset(self, deg, center, canvas_width, contours):
        re_theta = 45/self.theta_max
        #re_theta = deg/self.theta_max
        re_r, re_l = self.geometric_r_l(contours, center, canvas_width)
        return re_r, re_l, re_theta
    
    def visualize_tool(self, screen, display_point_arr, radii):

        gfxdraw.aapolygon(
            screen,
            [(display_point_arr[3][1], display_point_arr[3][0]), 
            (display_point_arr[2][1], display_point_arr[2][0]), 
            (display_point_arr[1][1], display_point_arr[1][0]),
            (display_point_arr[0][1], display_point_arr[0][0])], 
            (0,0,70),
        )

        gfxdraw.filled_polygon(
            screen,
            [(display_point_arr[3][1], display_point_arr[3][0]), 
            (display_point_arr[2][1], display_point_arr[2][0]), 
            (display_point_arr[1][1], display_point_arr[1][0]),
            (display_point_arr[0][1], display_point_arr[0][0])], 
            (239,70,70),
        )
        return screen