import gym, cv2
import pygame as G
import numpy as np

from pathlib import Path
from Env.object_env import *
from Env.utils_env import *

class Load(Parameters, gym.Env):
    """
    Description:
        게임환경 구성을 위한 클래스

    """
    def __init__(self):
        """
        Description:
            게임환경 구축을 위한 함수

        """
        super().__init__()
        self.clock = G.time.Clock()
        self.screen = Screen()
        self.target = Target()
        self.cursor = Cursor()
        self.event = Event()

        visible_mouse(False)

        self.root_dir = '/Users/imlim/Documents/Project/BRAIN_RL'
        self.path = np.load(Path(self.root_dir, 'Data', 'total_path.npy')).astype('int')
        self.done = False
        self.count = 0

        # Action space 정의
        act_high = 1.0

        self.action_r = gym.spaces.Box(low=np.float(0), high=np.float(act_high), shape=(1,))
        self.action_theta = gym.spaces.Box(low=np.float(-act_high), high=np.float(act_high), shape=(1,))

        # Observation space 정의
        obs_high = np.array([1920., 1080.], dtype=np.float)
        obs_low = np.array([0., 0.], dtype=np.float)
        self.observation_space = gym.spaces.Box(low=obs_low, high=obs_high, shape=(2,), dtype=np.float)

        # State space 정의
        self.state = np.array([self.init_x, self.init_y, self.init_x, self.init_y], dtype=np.float)

        self.action = None

    def step(self, r, theta):
        """
        Description:
            게임 진행을 위한 함수.
            
            Target 업데이트 | Cursor 업데이트 | Reward 계산 | State 업데이트

        """
        target = self.target.move(self.path, idx=self.count)

        self.action = np.array([r[0] * 6, theta[0] * 180])
        cursor = self.cursor.move('base', self.action)

        self.state = np.array([cursor[0], cursor[1], target[0], target[1]], dtype=np.float)

        distance = euclidean_distance(self.state)
        reward = distance_reward(distance, sigma=100)

        if (self.duration == self.count) or distance >= 350:
            self.done = True

        self.count += 1

        if self.count != 0 and self.count % 1500 == 0:
            self.cursor = Cursor()
        
        return self.state, reward, self.done
    
    def reset(self):
        """
        Description:
            self.done이 True면, 게임 초기화시키는 함수

        """
        self.count = 0
        self.done = False
        self.target = Target()
        self.cursor = Cursor()
        self.event = Event()

        self.state = np.array([self.init_x, self.init_y, self.init_x, self.init_y], dtype=np.float)
        return self.state

    def to_frame(self, width, height):
        """
        Description:
            게임 디스플레이를 np.array로 변환하는 함수

        """
        self.render()
        tmp = G.surfarray.array3d(self.screen.screen)
        
        # (W, H, C) -> (H, W, C)
        tmp = tmp.transpose((1, 0, 2))

        # 이미지 사이즈 조절
        # 1) Padding 2) Cropping 3) Resize
        img_pad = np.pad(tmp, ((448, 448), (448, 448), (0, 0)))
        cropped_img = img_pad[int(self.state[1]): int(self.state[1]) + 896, int(self.state[0]):int(self.state[0]) + 896, :]
        image = cv2.resize(cropped_img, (width, height), interpolation=cv2.INTER_AREA)

        # RGB scale -> Gray scale
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        return image

    def render(self):
        """
        Description:
            게임정보를 디스플레이 하기위한 함수
            
        """
        # If don't use this, the console window is not the same with the state.
        G.event.pump()

        # Set the hertz
        clock_tick(self.hertz)

        # Fill the window with black ground.
        self.screen.overwrite()

        # Display target.
        self.target.display('base', self.screen)

        # If hit, then the target will show red color.
        if hit(self.state):
            self.event.hit_target(self.screen, self.target)
        else:
            pass

        # Display cursor
        self.cursor.display(self.screen)

        # Update the console window.
        flip()
        





