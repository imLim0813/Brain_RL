import os, sys, cv2, pickle
import nibabel as nib
import pygame as G
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

from pathlib import Path
from pygame.rect import *
from PIL import Image
from nilearn import image
from nilearn.glm.first_level import make_first_level_design_matrix, FirstLevelModel
from nilearn.glm.first_level import compute_regressor
from nilearn.plotting import plot_design_matrix
from Anlz.utils_anlz import cal_theta, euclidean_distance


class Behavior_Anlz:
    """
    Description:
        사람의 행동 데이터를 이용하여 실험 이미지, 실험 비디오, trial별 타겟을 맞춘 정도 시각화하기 위한 클래스
    
    Args:
        subj_name : 피험자 번호
        root_dir : 최상위 경로

    """
    def __init__(self, subj_name: str, root_dir: Path):
        self.subj_name = subj_name
        self.root_dir = root_dir
        self.save_dir = Path(self.root_dir, 'Result', 'Human')
        
        self.orig_data = self.load_data()

    def load_data(self, run: str = 'All') -> None:
        """
        Description:
            .pkl 파일로 저장된 사람의 행동 데이터를 불러오는 함수.

        Args:
            run : run 번호.

        Returns:
            딕셔너리 형태의 사람 행동 데이터 (keys : ['cursor', 'target', 'joystick', 'hit', 'time'])

        """

        # load behavior data for a run.
        if run != 'All':
            with open(Path(self.root_dir, 'Data', 'Human', 'Behavior', self.subj_name, f'behavior_data_{run}.pkl'), 'rb') as f:
                data = pickle.load(f)    
            for name in data.keys():
                data[name] = np.array(data[name])
            # 35 : The difference between the center points of the cursor and target.
            data['cursor'] = np.array(data['cursor']) - 35

        # load all data.
        else:
            data = {}
            for key in ['cursor', 'target', 'joystick', 'hit', 'time']:
                data[key] = None

            for run in [i for i in range(1, 7, 1)]:
                run_data = self.load_data(run)
                for name, value in run_data.items():
                    if run == 1:
                        exec(f'data[name] = np.array(value)')
                    else:
                        if name != 'time':
                            exec(f'data[name] = np.concatenate([data[name], np.array(value)], axis=0)')
                        else:
                            exec(f'data[name] = np.concatenate([data[name], np.array(value) + 700 * (run-1)], axis=0)')
        return data

    def make_Video(self, run: str, trial: str) -> None:
        """
        Description:
            피험자가 진행한 실험을 동영상으로 만드는 함수.

        Args:
            run  : run 번호
            trial : trial 번호

        Returns:
            .mp4 파일

        """

        data_b = self.load_data(run)

        G.init()
        screen = G.display.set_mode([1920, 1080])
        frame = []   

        # 1465 : 한 trial의 frame 개수 | 20 : 하나의 run에 존재하는 trial의 개수
        for i in range(20 * 1465 * (run - 1) + 1465 * (trial - 1), 20 * 1465 * (run - 1) + 1465 * trial):
            screen.fill(G.Color(0, 0, 0))
            t_rect = Rect(int(data_b['target'][i][0]), int(data_b['target'][i][1]), 70, 70)
            G.draw.ellipse(screen, G.Color(128, 128, 128), t_rect, 0)

            if data_b['hit'][:-1][i][0] == 1:
                G.draw.ellipse(screen, G.Color(255, 0, 0), t_rect, 0)

            G.draw.line(screen, G.Color(255, 255, 255), (int(data_b['cursor'][:-1][i][0]) - 10 + 35, int(data_b['cursor'][:-1][i][1]) + 35),
                        (int(data_b['cursor'][:-1][i][0]) + 10 + 35, int(data_b['cursor'][:-1][i][1]) + 35), 3)
            G.draw.line(screen, G.Color(255, 255, 255), (int(data_b['cursor'][:-1][i][0]) + 35, int(data_b['cursor'][:-1][i][1] - 10 + 35)),
                        (int(data_b['cursor'][:-1][i][0]) + 35, int(data_b['cursor'][:-1][i][1] + 10 + 35)), 3)
            G.display.flip()
            G.image.save(screen, 'abc.BMP')
            png = Image.open('./abc.BMP')
            png.load()

            background = Image.new("RGB", png.size, (255, 255, 255))
            background.paste(png, mask=list(png.split())[3])
            frame.append(background)
        G.quit()

        frame_array = []
        for i in range(len(frame)):
            frame_array.append(cv2.cvtColor(np.array(frame[i]), cv2.COLOR_RGB2BGR))

        height, width, _ = frame_array[0].shape
        size = (width, height)

        save_path = Path(self.save_dir, self.subj_name, 'Video', 'Run {:02d}'.format(run))
        save_path.mkdir(parents=True ,exist_ok=True)

        out = cv2.VideoWriter(str(Path(save_path, 'Trial {:02d}.mp4'.format(trial))), fourcc=0x7634706d, fps=60, frameSize=size)

        for i in range(1465):
            out.write(frame_array[i])
        out.release()
        os.system('rm -rf ./abc.BMP')
        
        print('=' * 50)
        print(f'{self.subj_name} Video file is saved!')
        print('=' * 50)

        return True

    def make_Image(self, run: int, trial: int) -> None:
        """
        Description:
            피험자가 진행한 실험을 이미지로 만드는 함수

        Args:
            run  : run 번호
            trial : trial 번호

        Returns:
            .png 파일

        """

        data_b = self.load_data(run)
        G.init()
        screen = G.display.set_mode([1920, 1080])
        screen.fill(G.Color(255, 255, 255))
        color_list = [(159, 231, 249), (255, 153, 153)]

        for i in range(20 * 1465 * (run - 1) + 1465 * (trial - 1), 20 * 1465 * (run - 1) + 1465 * (trial)):
            G.event.pump()
            G.draw.circle(screen, G.Color(color_list[0]), (int(data_b['target'][i][0]), int(data_b['target'][i][1])), 3, 0)
            G.draw.circle(screen, G.Color(color_list[1]), (int(data_b['cursor'][i][0]), int(data_b['cursor'][i][1])), 3, 0)
            G.display.flip()

        string_image = G.image.tostring(screen, 'RGB')
        temp_surf = G.image.fromstring(string_image, (1920, 1080), 'RGB')
        tmp = G.surfarray.array3d(temp_surf)
        tmp = tmp.transpose((1, 0, 2))
        G.quit()

        plt.imshow(tmp)
        plt.axis('off')
        plt.text(20, 0, '- : target', c = [i / 255 for i in color_list[0]])
        plt.text(20, 70, '- : cursor', c = [i / 255 for i in color_list[1]])

        save_path = Path(self.save_dir, self.subj_name, 'Image', 'Run {:02d}'.format(run))
        save_path.mkdir(parents=True ,exist_ok=True)
        plt.savefig(str(Path(save_path, 'Trial {:02d}.png'.format(trial))), dpi=300)

        print('=' * 50)
        print(f'{self.subj_name} Image file is saved!')
        print('=' * 50)

        return True

    def make_LearningCurve(self, mode: str = 'Base') -> None:
        """
        Description:
            피험자가 타겟을 맞춘 정도를 trial별로 구한 후, 시각화하는 함수

        Args:
            mode : 추후 실험을 위해 ['Base', 'Adap']
            
        Returns:
            .jpg 파일

        """

        save_path = Path(self.save_dir, self.subj_name, 'Learning_Curve')
        save_path.mkdir(parents=True ,exist_ok=True)

        if mode == 'Base':
            hit_rate = []
            for trial in range(20 * 6):
                hit_sum = np.array(self.orig_data['hit'][1465 * trial: 1465 * (trial + 1)]).sum()
                hit_rate.append(float('{:.2f}'.format(hit_sum / 1465)))

            plt.rcParams["font.family"] = 'AppleGothic'
            plt.rcParams["font.size"] = 12
            plt.figure(figsize=(10, 5))
            plt.plot(np.arange(1, 121, 1), hit_rate, color='gray')

            for idx in range(0, 120, 20):
                plt.axvline([idx], color='#ff9999', linestyle=':', linewidth=1)

            plt.yticks(np.arange(0, 1.1, 0.1))
            plt.title('Learning rate ( {} )'.format(self.subj_name))
            plt.ylabel('Hit rate')
            plt.xlim(0, 120)
            plt.xlabel('Trial number')

            for idx, i in enumerate(range(7, 120, 20)):
                plt.text(i, 0.1, 'Run {}'.format(idx + 1))

            plt.savefig(str(Path(save_path, 'Learning_Curve({}).jpg'.format(mode))))

        if mode == 'Adap':
            hit_rate = []
            for trial in range(20 * 6):
                hit_sum = np.array(self.data['hit'][1465 * trial: 1465 * (trial + 1)]).sum()
                hit_rate.append(float('{:.2f}'.format(hit_sum / 1465)))
            hit_rate.append(hit_rate[-1])

            color_list = ['gray', '#ff9999', '#ff9999', 'gray', 'gray', '#ff9999', '#ff9999', 'gray', 'gray', '#ff9999',
                        '#ff9999', 'gray']

            plt.rcParams["font.family"] = 'AppleGothic'
            plt.rcParams["font.size"] = 12
            plt.figure(figsize=(10, 5))

            label_list = ['Adap', 'Base']

            for i in range(12):
                if i < 10:
                    plt.plot(np.arange(i * 10, 10 * (i + 1) + 1, 1), hit_rate[10 * i: 10 * (i + 1) + 1],
                            color=color_list[i])
                else:
                    plt.plot(np.arange(i * 10, 10 * (i + 1) + 1, 1), hit_rate[10 * i: 10 * (i + 1) + 1],
                            color=color_list[i], label=label_list[i % 2])

            for idx in range(0, 120, 10):
                plt.axvline([idx], color='#ff9999', linestyle=':', linewidth=1)

            plt.yticks(np.arange(0, 1.1, 0.1))
            plt.title('Learning rate ( {} )'.format(self.subj_name))
            plt.ylabel('Hit rate')
            plt.xlim(0, 120)
            plt.xlabel('Trial number')

            for idx, i in enumerate(range(7, 120, 20)):
                plt.text(i, 0.4, 'Run {}'.format(idx + 1))

            plt.legend(loc='lower right')

            plt.savefig(str(Path(save_path, 'Learning_Curve({}).jpg'.format(mode))))

        print('=' * 50)
        print(f'{self.subj_name} Learning Curve file is saved!')
        print('=' * 50)

        return True


class Data_Load:
    """
    Description:
        피험자가 진행한 실험을 수정하기 위한 클래스.
    
    Args:
        subj_name : 피험자 번호
        root_dir : 최상위 폴더

    """

    def __init__(self, subj_name: str, root_dir: str):
        self.subj_name = subj_name
        self.root_dir = root_dir
        self.orig = Behavior_Anlz(self.subj_name, self.root_dir).orig_data
        self.features = self.extract_features(self.orig)
        
        self.total = self.add_rest(self.orig)
        self.features_total = self.extract_features(self.total)

    def add_rest(self, orig_data: dict) -> dict:
        """
        Description:
            피험자 행동 데이터에 휴식시간(rest time)을 고려하기 위한 함수

        Args:
            orig_data : 행동 데이터

        Returns:
            휴식시간을 고려한 행동 데이터

        """
        data = {}
        for name in orig_data.keys():
            tmp = []

            if name == 'cursor' or name == 'target':
                rest_data = [960, 540]

            if name == 'joystick':
                rest_data = [0, 0]

            if name == 'hit':
                rest_data = [0]

            for i in range(1, orig_data[name].shape[0] + 1, 1):
                tmp.append(orig_data[name][i-1])

                if i % 1465 == 0 and name != 'time':
                    for _ in range(586):
                        tmp.append(rest_data)

                if i % 1465 == 0 and name == 'time':
                    for idx in range(1, 587, 1):
                        if i != 175800:
                            time_diff = orig_data[name][i] - orig_data[name][i-1]
                        if i == 175800:
                            time_diff = 4200 - orig_data[name][i-1]
                        tmp.append(orig_data[name][i - 1] + time_diff * idx / 586)
            data[name] = np.array(tmp)

        return data

    def extract_features(self, data: dict) -> dict:
        """
        Description:
            행동 데이터로부터 feature를 추출하기 위한 함수.

        Args:
            data : 행동 데이터
        
        Returns:
            feature의 이름을 key, 그 값을 value로 가지는 딕셔너리

            distance : 타겟과 커서 간의 차이 (x, y)
            euclidean : 타겟과 커서 간의 유클리디안 거리
            joystick : 조이스틱의 데카르트 좌표값
            polar : 조이스틱의 극 좌표값
            hit : 타겟을 맞췄는지에 대한 여부
            time : 시간

        """
        features = {}
        state = np.concatenate([data['cursor'], data['target']], axis=1)
        features['distance'] = data['target'] - data['cursor']
        features['euclidean'] = []
        for idx in range(state.shape[0]):
            features['euclidean'].append(euclidean_distance(state[idx]))
        features['euclidean'] = np.array(features['euclidean']).reshape(-1, 1)
        features['joystick'] = data['joystick']
        features['polar'] = []
        for idx in range(features['joystick'].shape[0]):
            features['polar'].append(
                [np.sqrt(data['joystick'][idx][0] ** 2 + data['joystick'][idx][1] ** 2), cal_theta(data['joystick'], idx)])
        features['polar'] = np.array(features['polar'])
        features['hit'] = data['hit']
        features['time'] = data['time']

        return features

