import pygame as G
import numpy as np
from Env.utils_env import Parameters
from Env.utils_env import base, adapt, reverse
from pygame.rect import *


class Cursor(Parameters):
    """
    Description:
        게임 내 커서에 관한 클래스

    """
    def __init__(self):
        super().__init__()
        
        # A cursor's cartesian coordinate values.
        self.x = self.init_x
        self.y = self.init_y

        # Prevent a cursor from moving out of the display console.
        self.max_x = self.width - self.target_diameter
        self.max_y = self.height - self.target_diameter

    def display(self, screen):
        """
        Description:
            데카르트 좌표 x, y 값을 이용하여 커서를 시각화하기 위한 함수.

            주의 ) 중심점에서 + 35씩 되어있음. (커서와 타겟의 중앙을 맞추기 위함)

        Args:
            screen : 커서가 출력될 디스플레이

        """
        G.draw.line(screen.screen, self.white, (self.x + 25, self.y + 35), (self.x + 45, self.y + 35), 3)
        G.draw.line(screen.screen, self.white, (self.x + 35, self.y + 25), (self.x + 35, self.y + 45), 3)

    def move(self, mode: str, action):
        """
        Description:
            커서의 x, y 값을 업데이트 시키는 함수
        
        Args:
            mode : base | adapt | reverse
            action : DRL 모델 action 모듈의 출력 값.

        Return:
            업데이트 이후 x, y 값
        """
        # Convert polar coordinate to cartesian coordinate.
        act_x = action[0] * np.cos(np.deg2rad((action[1])))
        act_y = action[0] * np.sin(np.deg2rad(action[1]))

        # Update the cursor coordinate.
        if mode == 'base':
            self.x, self.y = base(self.x, self.y, act_x, act_y, self.max_x, self.max_y)
        elif mode == 'adapt':
            self.x, self.y = adapt(self.x, self.y, act_x, act_y, self.max_x, self.max_y)
        elif mode == 'reverse':
            self.x, self.y = reverse(self.x, self.y, act_x, act_y, self.max_x, self.max_y)
        else:
            print('Choose the mode among...')
            RuntimeError()

        return [int(self.x), int(self.y)] # When drawing the cursor, pygame requires int values.


class Target(Parameters):
    """
    Description:
        게임 내 타겟에 관한 클래스

    """
    def __init__(self):
        super().__init__()
        self.rect = Rect(self.init_x, self.init_y, self.target_diameter, self.target_diameter)
        self.target = None

    def move(self, path, idx):
        """
        Description:
            커서의 x, y 값을 업데이트 시키는 함수
        
        Args:
            path : 타겟 경로를 값으로 가지는 데이터
            index : framestep

        Return:
            업데이트 이후 x, y 값

        """
        self.rect = Rect(path[idx][0], path[idx][1], self.target_diameter, self.target_diameter)

        if self.rect.left < 0:
            self.rect.left = 0
        if self.rect.right > self.width:
            self.rect.right = self.width
        if self.rect.top < 0:
            self.rect.top = 0
        if self.rect.bottom > self.height:
            self.rect.bottom = self.height

        return [list(self.rect.copy())[0], list(self.rect.copy())[1]]

    def display(self, mode, screen):
        """
        Description:
            데카르트 좌표 x, y 값을 이용하여 타겟을 시각화하기 위한 함수.

        Args:
            mode : base | adapt | reverse 
            screen : 커서가 출력될 디스플레이

        """
        if mode == 'base':
            self.target = G.draw.ellipse(screen.screen, self.gray, self.rect, 0)
        elif mode == 'adapt':
            self.target = G.draw.ellipse(screen.screen, self.blue, self.rect, 0)
        elif mode == 'reverse':
            self.target = G.draw.ellipse(screen.screen, self.yellow, self.rect, 0)