import os
import math
from typing import Optional
import cv2
import numpy as np
import gym
from gym import spaces

from gym.envs.classic_control import utils
from gym.error import DependencyNotInstalled
from gym.utils import seeding
from gym.utils.renderer import Renderer


import  Callienv.envs.skel_utils as skel_utils
'''
state: [period, r, l, theta, curvature, r_prime, vec_x, vec_y]

        period: [0,1]
            current position / one stroke length  
            [notice the length of skelecton list includes multiple strokes, here the period is stroke-based]

        r, l, theta: [0,1]
            geometric properties.
            eg: Droplet brush model: r indicates radius of the circle, l indicates vrush tip length. Theta indicates the rotation angle.
                Ellipse shape brush: r indicates semi-major axis and l indicates semi-minor axis.  Theta indicates the rotation angle.
                Chisel Tip Marker:   r indicates semi-length, and l indicates semi-width(quadrangle shape!). Theta indicates the rotation angle.

        curvature: [0,1]
            current curvature (calculated by sin() function)
        
        r_prime: [0,1]
            The distance from the original point to the point after moving in the previous step.  (may change to vectorizes further)
        
        vec_x, vec_y: [0,1]
            Future direction for the current stroke.
            Specifically, we compute the unit vector between the future point (x_prime, y_prime) and the current point (x,y).
            x_prime - x, y_prime - y
        
        total: 8

action: [r_prime, theta_prime]

        r_prime: [-1,1] (multiply r_prime_bound)
            in renderer, it will multiply a max_dist
            We move the original point with this distance by a polar distance.
            
        theta_prime: [-1,1] (multiply pi)
            We move the original point with this angle by a polar distance.

        total: 2

reward: 
        in step function we'll calculate new aoto-fit r_new,l_new,theta_new and so on

        -2* math.abs(r_prime_new)*curvature / 0.4* cos_sim of theta and new_theta + 0.6
'''

class CalliEnv(gym.Env):
    metadata = {
        "render_modes": ["human", "rgb_array", "single_rgb_array"],
        "render_fps": 30,
    }

    def __init__(self, tool, folder_path: str, output_path: str, visualize_path: str, env_num: int, env_rank:tuple, render_mode: Optional[str] = None, graph_width = 256,\
                 screen_width = 512, canvas_width = 300, image_iter = 20, start_update = 5, update = 5, ema_gamma = 0.95): # 40 10 5
        '''
        # modify canvas width!
        tool : class defined in tool.py
        notice that may be misused in draw_canvas !!!
        '''
        self.tool = tool
        ## action space
        self.r_prime_low = -1
        self.r_prime_high = 1
        self.theta_prime_low = -1
        self.theta_prime_high = 1
        self.rotate_prime_low = -1
        self.rotate_prime_high = 1
        self.r_prime_bound = 0.022  #control freq: can be modified
        self.rotate_bound = 10000

        ## observation space
        self.period_min = 0
        self.period_max = 1
        self.r_min = self.tool.r_min
        self.r_max = self.tool.r_max
        self.l_min = self.tool.l_min
        self.l_max = self.tool.l_max
        self.theta_min = self.tool.theta_min
        self.theta_max = self.tool.theta_max
        self.theta_step  = self.tool.theta_step
        self.curv_min = 0
        self.curv_max = 1
        self.vec_x_min = -1
        self.vec_x_max = 1
        self.vec_y_min = -1
        self.vec_y_max = 1

        ## image and coarse skelecton directory pool
        self.folder_path = folder_path
        self.output_path = output_path
        self.visualize_path = visualize_path
        self.env_num = env_num   #how many environs
        self.env_rank = env_rank #tuple, (ith,total), eg: (0,10), (1,10) ...(9,10)
        self.counter = 0 # when counter % image_iter ==0, update image; and pick counter//image_iter th tuple
        self.update = update
        self.image_iter = image_iter
        self.start_update = start_update
        self.ema_gamma = ema_gamma

        self.data_pool = [] # list of tuple
        ## renderer setting
        self.render_mode = render_mode
        self.renderer = Renderer(self.render_mode, self._render)
        self.graph_width = graph_width
        self.screen_width = screen_width
        self.canvas_width = canvas_width
        self.screen = None
        self.clock = None
        self.isopen = True
        
        ## state action space
        if self.tool.action_space == 2:
            self.actlow = np.array([self.r_prime_low, self.theta_prime_low], dtype=np.float32)
            self.acthigh = np.array([self.r_prime_high, self.theta_prime_high], dtype=np.float32)

        elif self.tool.action_space == 3:
            self.actlow = np.array([self.r_prime_low, self.theta_prime_low, self.rotate_prime_low], dtype=np.float32)
            self.acthigh = np.array([self.r_prime_high, self.theta_prime_high, self.rotate_prime_high], dtype=np.float32)
        
        self.obslow = np.array([self.period_min, self.r_min, self.l_min, 0, self.curv_min,\
                                self.r_prime_low,self.vec_x_min,self.vec_y_min], dtype=np.float32)
        self.obshigh = np.array([self.period_max,self.r_max, self.l_max, 1, self.curv_max,\
                                self.r_prime_high, self.vec_x_max, self.vec_y_max], dtype=np.float32)
        
        self.action_space = spaces.Box(self.actlow, self.acthigh, dtype=np.float32)
        self.observation_space = spaces.Box(self.obslow, self.obshigh, dtype=np.float32)  

        self.fill_skel_and_img_pool('.npy')

    def fill_skel_and_img_pool(self, datatype):
        self.tot = int(len(os.listdir(self.folder_path))/2)
        delta = int(self.tot / self.env_num)
        self.start_idx = int(self.env_rank[0]/self.env_rank[1] * self.tot)
        self.end_idx = int((self.env_rank[0] + delta)/self.env_rank[1] * self.tot)
        self.list_num = self.end_idx - self.start_idx
        
        self.data_pool = [(self.folder_path+str(idx)+".png", self.folder_path+str(idx)+datatype)\
                           for idx in range(self.start_idx, self.end_idx)]
        
        self.control_list = [np.array([]) for z in range(self.list_num)]
        self.origin_list = [np.array([]) for z in range(self.list_num)]

    def load_skel_and_img(self, pick_data, count):
        '''
        New Notice: after consideration, we decided to call this function in the reset() method to cater to Tianshou structure
        
        Crucial function called before reset. 
        pt_list records the result of pen state recorded from Coarse sequence extraction.
            pt=1 means a new start of a stroke
        self.pt_idx is an array of indexes (represent where pt=1)
        self.pt_indices records the distance between two pt=1 states. 
            1/self.pt_indices[i] represents the scale of period for each step in each stroke.
            if pt=1 after a step() call, it means a new scale (1/pt_indices[i+1]) should be adopted.
        '''
        img_path, skel_path = pick_data
        self.img_path = img_path       #renderer
        self.stroke_img = cv2.imread(self.img_path, 0)
        
        if self.stroke_img.shape != (self.graph_width, self.graph_width):
            self.stroke_img = cv2.resize(self.stroke_img, ((self.graph_width, self.graph_width)))
            assert self.stroke_img.shape == (self.graph_width, self.graph_width)
        self.stroke_downsample = cv2.threshold(cv2.resize(self.stroke_img,(self.canvas_width,self.canvas_width))\
                                               ,127,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)[1]/255
        
        self.record_canvas = np.zeros((self.canvas_width,self.canvas_width))

        _, thresh = cv2.threshold(self.stroke_img, 127,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
        self.contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

        skel_list = skel_utils.transfer_data(skel_path)
        skel_list = skel_utils.add_beg_end_seq(skel_list, self.contours, self.graph_width)
        self.origin_list[count] = np.copy(skel_list)
        self.skel_list = skel_list[:,1:]
        self.new_skel_list = np.array([])
        self.r_list = np.array([])
        self.R_list = np.array([])
        self.pt_list = skel_list[:,0]
        if self.pt_list[0] == 0:
            cur = np.append(np.array([0]), np.where(self.pt_list==1))
        else:
            cur = np.where(self.pt_list==1)
        self.pt_idx = np.append(cur, np.array([self.skel_list.shape[0]]))
        self.pt_indices = np.diff(self.pt_idx,n=1)
        assert self.pt_indices.shape[0] >=1
        
    
    def calc_four_points(self,state): 
        ''' x,y,r,l,theta -> (x,y), (x_rgtintsec, y_rgtintsec), (x_tip, y_tip), (x_lftintsec, y_lftintsec)
            params: state: [period, r, l, theta, curvature, r_prime, vec_x, vec_y] 
            NOTICE: 
                in writing brush, cur_r and cur_l are real pixels/image pixels in [0,1].
                however, in elipse and chisel marker the r and l are almost fixed, 
                meaning in reset() and step() we should provide real pixel/img width!!!!
        ''' 
        
        _, cur_r, cur_l, cur_theta, _, cur_r_prime, cur_vec_x, cur_vec_y = state
        #resize
        current = self.skel_list_cnter
        cur_r_prime *= self.r_prime_bound
        cur_theta *= self.theta_max

        #circle center
        vec_rot = np.array([cur_vec_y,-cur_vec_x])
        v_1 = self.skel_list[current] + cur_r_prime * vec_rot  #center

        rad = math.radians(cur_theta) # counterclockwise, [0, 2 pi]
        return self.tool.calc_four_points(v_1, cur_r, cur_l, rad)  #all in [0,1]
    
    def draw_canvas(self, four_points, cur_r):
        """
        Draw a 4-dimensional array (the first element is the center of a circle and the remaining
        three elements are the vertices of a triangle) onto a 128x128 canvas. The array is scaled
        up by a factor of 128. The function takes in the current radius of the circle and draws
        the circle and triangle onto the canvas.
        """
        canvas_width = self.canvas_width
        r = int(cur_r*canvas_width)
        if r == 0:
            return
        rr, cc = self.tool.draw_canvas(canvas_width, four_points, cur_r)
        self.record_canvas[rr,cc] = 1

    def step(self, action: np.ndarray): 
        
        # receive action
        next_r_prime = action[0]
        '''
        action dim = 2 r, theta
        however, in different tools like marker, brush dimension is 3 (r theta, rotate)
        '''
        next_theta_prime = action[1]
        next_theta_prime = next_theta_prime*math.pi # [-pi to pi]

        self.skel_list_cnter +=1

        next_state_0 = self.state[0]+1/self.pt_indices[self.i]

        ### calculate next step
        ## calc period, curvature, r_prime, vec_x, vec_y
        if self.skel_list_cnter >= self.pt_idx[self.i+1]-1:  ## when self.skel_list_cnter == self.i, wrong vector
            next_vec_x, next_vec_y = self.state[-2], self.state[-1]
        else:
            vec = self.skel_list[self.skel_list_cnter-1] - self.skel_list[self.skel_list_cnter+1]
            next_vec_x, next_vec_y = vec/np.linalg.norm(vec)
            
        ## calculate next midpoint of the tip
        delta_x = -math.sin(next_theta_prime)*next_r_prime*self.r_prime_bound
        delta_y = math.cos(next_theta_prime)*next_r_prime*self.r_prime_bound
        next_x = self.skel_list[self.skel_list_cnter][0]+delta_x
        next_y = self.skel_list[self.skel_list_cnter][1]+delta_y
        next_x, next_y = max(min(0.99,next_x),0), max(min(0.99,next_y),0)

        ## next_x, next_y append in self.new_skel_list
        self.new_skel_list = np.concatenate((self.new_skel_list, np.array([next_x, next_y])))

        next_center_in_graph = np.array([int(next_x*self.graph_width),int(next_y*self.graph_width)])  #256

        ## calculate curvature
        if self.skel_list_cnter >= self.pt_idx[self.i+1]-2:  # a stroke ends
            next_curv = 1
        else:
            futurevec = self.skel_list[self.skel_list_cnter+1] - self.skel_list[self.skel_list_cnter]
            next_curv = math.sin(math.acos(np.inner(vec, futurevec)/np.linalg.norm(vec)*np.linalg.norm(futurevec))/2)
        
        ## calculate dynamics (r, l, theta)
        next_r, next_l, next_theta =  self.tool.dynamics(self.state, action, next_center_in_graph,\
                                                         next_vec_x, next_vec_y, self.contours, self.graph_width) ### CAUTION: GRAPH WIDTH IS 256!!!

        ## re-calculate next_theta, next_vec_x and next_vec_y.
        if next_state_0 >= 0.999:
            next_state_0 = 0
            self.i +=1
            vec = self.skel_list[self.skel_list_cnter]-self.skel_list[self.skel_list_cnter+2]
            vec = vec/np.linalg.norm(vec)
            next_theta = math.degrees(math.acos(np.inner(vec,np.array([0,1])))) #[0-180]
            if np.inner(vec,np.array([1,0])) >0:  #sin >0
                next_theta = self.theta_max - next_theta
            next_vec_x, next_vec_y = vec

        self.r_list = np.concatenate((self.r_list, np.array([next_r])))
        
        # calculate terminated and done
        # 如果一个字走完一遍，那么terminated = True，也就是reset(), 对应天授里面logical_or(term, trunc) = done, done就reset是对的
        terminated, done = False, False
        if self.skel_list_cnter == self.skel_list.shape[0]-2 or self.i == self.pt_indices.shape[0]:
            terminated = True
        
        #update state
        last_r = self.state[1]
        self.state = np.array([next_state_0, next_r, next_l, next_theta/self.theta_max,\
                                next_curv, next_r_prime, next_vec_x, next_vec_y],dtype=np.float32)
        # calculate reward
        reward = self.calc_reward(next_r_prime, last_r, terminated)
        
        diff = self.counter - (self.counter // self.image_iter)*self.image_iter

        if terminated:
            if diff >= self.start_update and diff % self.update == 0:
                self.skel_list = skel_utils.EMA(self.skel_list, self.new_skel_list.reshape(-1, 2), self.ema_gamma)
  
            self.new_skel_list = np.empty(0)
            self.R_list = np.concatenate((self.R_list, np.array([self.r_list[0]/2])))
            self.R_list = np.copy(self.r_list)
            self.r_list = np.empty(0)
        
        #return self.state, reward, terminated, {}
        # shimmy tianshou/env/venvs.py/patch_env_generator/patched need to be modified!!
        return self.state, reward, terminated, done, {}
    
    def seed(self, seed):
        if seed is not None:
            self._np_random, seed = seeding.np_random(seed)

    def reset(self,
        *,
        seed: Optional[int] = None,
        return_info: bool = False,
        options: Optional[dict] = None):
        ''' 
        self.i is a variable that indicates which stroke is currently being drawn in a character.
        NOTICE: In order to adapt to the Tianshou architecture, we maintain a list in the environment
                that stores a binary tuple of images and coarse stroke arrays. 
                For each element in the list, we set a repetition count and iterate through the list.
        '''
        if seed is not None:
            self._np_random, seed = seeding.np_random(seed)
        
        self.screen = None
        self.skel_list_cnter = 0
        self.i = 0

        self.record_canvas = np.zeros((self.canvas_width,self.canvas_width))
        re_period = 0
        re_r_prime = 0

        if self.counter % self.image_iter == 0: #record image-wise
            ## change environment imgs and skels
            
            if self.counter !=0:
                last_id = ((self.counter-1) //self.image_iter) % self.list_num

                self.control_list[last_id] = np.hstack((self.pt_list[:, np.newaxis],self.skel_list))
                r_len = self.R_list.shape[0]

                self.control_list[last_id] = np.hstack((self.control_list[last_id][:r_len, :], self.R_list[:,np.newaxis])) # 4 dim
                self.R_list = np.empty(0)
                
                last_count = ((self.counter-1) // self.image_iter) % self.list_num
                save_control_dir = os.path.join(self.output_path, str(self.start_idx+last_count)+'_'+str(self.counter)+'.npy')
                np.save(save_control_dir, self.control_list[last_id])
                
                if self.visualize_path is not None:
                    save_vis_dir = os.path.join(self.visualize_path, str(self.start_idx+last_count)+'_'+str(self.counter)+'.png')
                    skel_utils.save_visualization(save_vis_dir,self.data_pool[last_count][0], self.control_list[last_id], None)
 
            count = (self.counter // self.image_iter) % self.list_num
            pick_data = self.data_pool[count]
            self.load_skel_and_img(pick_data, count)
        self.counter+=1

        vec = self.skel_list[0]-self.skel_list[1]
        deg = math.degrees(math.acos(np.inner(vec,np.array([0,1]))/np.linalg.norm(vec))) #[0-180]
        if np.inner(vec,np.array([1,0])) >0:  #sin >0
            deg = self.theta_max - deg

        re_vec = vec
        re_vec_x, re_vec_y = re_vec/np.linalg.norm(re_vec)
        center_in_graph = np.array([int(self.skel_list[0][0]*self.graph_width),int(self.skel_list[0][1]*self.graph_width)])
        
        re_r, re_l, re_theta = self.tool.reset(deg, center_in_graph,\
                                                self.canvas_width, self.contours)
        re_curv = 1
        
        self.state = [re_period, re_r, re_l, re_theta, re_curv, re_r_prime, re_vec_x, re_vec_y]
        self.state = np.array(self.state, dtype=np.float32)

        
        '''
        dummyvectorenv里面实现了对每个avail的render,
        这里面的环境还需要再包一层Wrapper, 如果录像or可视化的话.
        好处就是,train以后可以不保存vids也不可视化,只需要不调整fps就好
        '''
        
        if not return_info:
            return self.state
        else:
            return self.state, {}
    def calc_reward(self, next_r_prime, last_r, terminated):
        '''
        IN_STEP:
            1. stroke_size
                beginning & end of each stroke: not expect large stroke (smaller better, not too small)
                center of each stroke: expect large as possible
            2. smoothness
            
            3. out of border penalty (in 1.)

            sum should in [-1, 1], not too much!!!
        TERMINATE:
            calculate the symmetric difference of pixels,normalize and map to [-60, 40]

        '''
        reward = 0
        next_period, next_r = self.state[0], self.state[1]
        points, cur_r = self.calc_four_points(self.state)
        if cur_r==0:
            cur_r +=1e-4

        rr, cc = self.tool.draw_canvas(self.canvas_width, points, cur_r)
        self.record_canvas[rr,cc] = 1

        tip_ds = self.stroke_downsample[rr,cc]
        tip_pix = tip_ds.shape[0]
        assert tip_pix >0
        stk_pix = np.sum(tip_ds)
        
        reward -= 0.08* (1/(math.sqrt(next_r+1e-4))-0.5/math.sqrt(self.r_max))**1.5
        
        ## terminate reward:
        points, cur_r = self.calc_four_points(self.state)
        self.draw_canvas(points, cur_r)

        if terminated:

            total_area = self.stroke_downsample.sum()
            rec = (self.record_canvas - self.stroke_downsample).flatten()
            
            rec = (np.inner(rec, rec)/total_area)
            reward += 80*(-(rec**0.4 - 0.5) + 0.2)

        return reward

    def render(self, mode="human"):
        if self.render_mode is not None:
            self.renderer.reset()
            self.renderer.render_step()
            return self.renderer.get_renders()
        ##record_video是wrapper, video_recorder里面写到，取的是这个frames的[-1]， 因此只需要返回一帧就好了
        
        else:
            return self._render(mode)

    def init_window(self):
        try:
            import pygame
            from pygame import gfxdraw
        except ImportError:
            raise DependencyNotInstalled(
                "pygame is not installed, run `pip install gym[classic_control]`"
            )
        self.screen.fill((255,255,255))
        
        plot = np.transpose(cv2.resize(self.stroke_img,(self.screen_width, self.screen_width)),(1,0))
        bg_surf = pygame.surfarray.make_surface(
                cv2.cvtColor(plot, cv2.COLOR_GRAY2RGB))
                
        bg_surf.set_alpha(50)

        self.screen.blit(bg_surf,(0,0))


    def _render(self, mode="rgb_array"):

        assert mode in self.metadata["render_modes"]
        try:
            import pygame
            from pygame import gfxdraw
        except ImportError:
            raise DependencyNotInstalled(
                "pygame is not installed, run `pip install gym[classic_control]`"
            )

        if self.screen is None:
            pygame.init()
            if mode == "human":
                pygame.display.init()
                self.screen = pygame.display.set_mode((self.screen_width, self.screen_width))
            else:
                self.screen = pygame.Surface((self.screen_width, self.screen_width))

            self.screen.fill((255,255,255))

            plot = np.transpose(cv2.resize(self.stroke_img,(self.screen_width, self.screen_width)),(1,0))
            bg_surf = pygame.surfarray.make_surface(
                cv2.cvtColor(plot, cv2.COLOR_GRAY2RGB)
            )
            bg_surf.set_alpha(80)

            self.screen.blit(bg_surf,(0,0))


        if self.clock is None:
            self.clock = pygame.time.Clock()

        point_arr, cur_r = self.calc_four_points(self.state)

        display_point_arr = (point_arr*self.screen_width).astype(np.int16)
        Radii = int(cur_r*self.screen_width)
        
        self.screen = self.tool.visualize_tool(self.screen, display_point_arr, Radii)
        
        if mode == "human":
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            pygame.display.flip()

        elif mode in {"rgb_array", "single_rgb_array"}:
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )
        
    def get_keys_to_action(self):
        return {(): 8, (276,): 0, (275,): 1}

    def close(self):
        if self.screen is not None:
            import pygame
            pygame.display.quit()
            pygame.quit()
            self.isopen = False
