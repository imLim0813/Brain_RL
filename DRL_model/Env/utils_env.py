import os
import pygame as G
import numpy as np
import matplotlib.pyplot as plt


class Parameters:
    """
    Description:
        여러 클래스에 거쳐 사용될 파라미터를 정리해놓은 클래스

    """
    def __init__(self):
        self.red = G.Color(255, 0, 0)
        self.gray = G.Color(128, 128, 128)
        self.green = G.Color(0, 255, 0)
        self.white = G.Color(255, 255, 255)
        self.black = G.Color(0, 0, 0)
        self.blue = G.Color(0, 0, 255)
        self.yellow = G.Color(255, 255, 0)

        self.width = 1920
        self.height = 1080
        self.cursor_diameter = 20
        self.target_diameter = 70

        self.hertz = 60
        self.time = 25
        self.trial = 20
        self.duration = self.hertz * self.time * self.trial

        self.init_x = 960
        self.init_y = 540


class Screen(Parameters):
    """
    Description:
        게임 콘솔 창 정의를 위한 클래스
        
    """
    def __init__(self):
        super().__init__()
        G.init()
        self.screen = G.display.set_mode([self.width, self.height])

    def overwrite(self):
        # 콘솔 창을 검은색으로 덮어쓰기
        self.screen.fill(self.black)


class Event(Parameters):
    """
    Description:
        이벤트 정의를 위한 클래스
        
    """

    def __init__(self):
        super().__init__()

    def hit_target(self, screen, target):
        # 커서가 타겟을 맞추면 타겟의 색을 빨간색으로 변경
        G.draw.ellipse(screen.screen, self.red, target.rect, 0)


def visible_mouse(tmp=True):
    """
    Description:
        콘솔 창에서 마우스 보기 여부
        
    """
    G.mouse.set_visible(tmp)


def clock_tick(time):
    """
    Description:
        FPS 설정에 관한 함수

    """
    clock = G.time.Clock()
    clock.tick(time)


def plot_image(frame, width=84, height=84):
    """
    Description:
        강화학습 모델의 입력 이미지를 시각화하기 위한 함수
        
    """
    frame = frame.reshape(1, width, height).transpose((2, 1, 0))
    plt.imshow(frame, cmap='gray')
    plt.show()


def distance(state):
    """
    Description:
        타겟과 커서간의 좌표 값 차이를 측정하는 함수
        
    """
    return np.array([(state[0] - state[2]), (state[1] - state[3])], dtype=np.float32)


def euclidean_distance(state):
    """
    Description:
        타겟과 커서간의 유클리디안 거리를 측정하는 함수
        
    """
    return np.sqrt((state[2] - state[0]) ** 2 + (state[3] - state[1]) ** 2)


def distance_reward(distance, sigma):
    """
    Description:
        타겟과 커서 간 거리를 이용하여 보상함수를 정의하기 위한 함수
        
    """
    return np.exp(-((distance ** 2) / (2 * (sigma ** 2))))


def hit(state):
    """
    Description:
        커서가 타겟을 맞췄는지 여부를 판단하기 위한 함수

    """
    return state[2] <= state[0] +35 <= state[2] + 70 \
           and state[3] <= state[1] + 35 <= state[3] + 70 \
           and euclidean_distance(state) <= 35


def flip():
    """
    Description:
        콘솔 창의 프레임 전환을 위한 함수

    """
    # 
    G.display.flip()


def base(cur_x, cur_y, act_x, act_y, max_x, max_y):
    # 커서의 base 환경을 위한 함수
    # 조이스틱과 커서의 움직임이 직관적으로 대응한다.
    cur_x += act_x
    cur_y += act_y

    if cur_y <= 0:
        cur_y = 0
    elif cur_y >= max_y:
        cur_y = max_y

    if cur_x <= 0:
        cur_x = 0
    elif cur_x >= max_x:
        cur_x = max_x

    return cur_x, cur_y


def adapt(cur_x, cur_y, act_x, act_y, max_x, max_y, degree=-90):
    # 커서의 adaptation 환경을 위한 함수
    # 조이스틱을 오른쪽으로 움직이면 커서는 위로 움직인다. ( 90 degree counterclockwise)
    prev_x = act_x
    prev_y = act_y

    if prev_x > 0:
        theta_final = degree + np.rad2deg(np.arctan(prev_y / prev_x))
    elif prev_x < 0:
        theta_final = 180 + degree + np.rad2deg(np.arctan(prev_y / prev_x))

    prev_x_rot = np.sqrt(prev_x ** 2 + prev_y ** 2) * np.cos(np.deg2rad(theta_final))
    prev_y_rot = np.sqrt(prev_x ** 2 + prev_y ** 2) * np.sin(np.deg2rad(theta_final))

    cur_x += prev_x_rot
    cur_y += prev_y_rot

    if cur_y <= 0:
        cur_y = 0
    elif cur_y >= max_y:
        cur_y = max_y

    if cur_x <= 0:
        cur_x = 0
    elif cur_x >= max_x:
        cur_x = max_x

    return cur_x, cur_y


def reverse(cur_x, cur_y, act_x, act_y, max_x, max_y, degree=0):
    # 커서의 reverse 환경을 위한 함수
    # 조이스틱이 45도를 기준으로 대칭적으로 움직인다.
    prev_x = act_x
    prev_y = act_y

    cur_x -= prev_y
    cur_y -= prev_x

    if cur_y <= 0:
        cur_y = 0
    elif cur_y >= max_y:
        cur_y = max_y

    if cur_x <= 0:
        cur_x = 0
    elif cur_x >= max_x:
        cur_x = max_x

    return cur_x, cur_y
