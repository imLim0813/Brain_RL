import glob, matplotlib
import matplotlib.pyplot as plt
import numpy as np

from pathlib import Path
from Anlz.utils_anlz import cal_theta, euclidean_distance

class Policy_Anlz:
    """
    Description:
        행동 정책을 시각화하기 위한 클래스

    Args:
        subj_name : 피험자 번호
        root_dir : 최상위 경로
        data : feature 행동 데이터

    """
    def __init__(self, subj_name: str, root_dir: str, data: dict):
        self.subj_name = subj_name
        self.root_dir = root_dir
        self.save_dir = Path(self.root_dir, 'Result', 'Policy')
        self.data = data
        self.corr_dict = {}

    def Human_Policy(self):
        """
        Description:
            사람 행동 데이터로부터 radian 분포, theta 분포, 그리고 행동정책 분포를 시각화하기 위한 함수

        """
        save_path = Path(self.save_dir, self.subj_name)
        save_path.mkdir(parents=True ,exist_ok=True)

        # Radian distribution
        plt.rc('font', size=8)
        plt.hist(self.data.features['polar'][:, 0])
        plt.title('Human_r')
        plt.xlim(0, 6)
        plt.savefig(Path(save_path, 'RADIAN.jpg'), dpi=300)
        plt.close('all')

        # Theta distribution
        plt.rc('font', size=8)
        plt.xlim(0, 361)
        plt.hist(self.data.features['polar'][:, 1])
        plt.title('Human_theta')
        plt.savefig(Path(save_path, 'THETA.jpg'), dpi=300)
        plt.close('all')

        # Policy distribution
        plt.rc('font', size=8)
        r = self.data.features['polar'][:, 0]
        theta = self.data.features['polar'][:, 1]
        rad = np.deg2rad(theta)

        rbins = np.linspace(0, 6, 13)
        abins = np.linspace(0, 2 * np.pi, 37)

        hist, _, _ = np.histogram2d(rad, r, bins=(abins, rbins))
        A, R = np.meshgrid(abins, rbins)
        self.corr_dict['Human'] = hist.T.reshape(-1, 1)

        fig, ax = plt.subplots(subplot_kw=dict(projection='polar'), figsize=(8, 8))
        normalize = matplotlib.colors.LogNorm()
        pc = ax.pcolormesh(A, R, hist.T, cmap='Oranges', norm=normalize)
        fig.colorbar(pc)
        ax.grid(True)
        plt.savefig(Path(save_path, 'Policy.jpg'), dpi=300)
        plt.close('all')

        print("=" * 50)
        print("The Human's Policy histogram has been saved!")
        print("=" * 50)

        return True # r, theta, hist

    def Target_policy(self):
        """
        Description:
            타겟 데이터로부터 radian 분포, theta 분포, 그리고 행동정책 분포를 시각화하기 위한 함수

        """

        save_path = Path(self.save_dir, 'Target')
        save_path.mkdir(parents=True ,exist_ok=True)

        target_data = np.load(Path(self.root_dir, 'Data', 'total_path.npy'))
        behav_data = {'target': []}

        # Rearrange target trajectory. (Participants saw 1465 frames per trial)
        for i in range(20):
            for idx in range(1500 * i, 1500 * i + 1465):
                behav_data['target'].append(target_data[idx])
        behav_data['target'] = np.array(behav_data['target'])

        # Calculate a target's action.
        target_action = []
        for idx in range(0, behav_data['target'].shape[0] - 1, 1):
            tmp = behav_data['target'][idx + 1] - behav_data['target'][idx]
            target_action.append(tmp)
        target_action.append([0, 0])
        target_action = np.array(target_action)

        data = {'joystick': target_action}

        # Transform a cartesian coordinate system to polar coordinate system.
        tmp = {'joystick': data['joystick'], 'polar': []}
        for idx in range(tmp['joystick'].shape[0]):
            # Action values are [0, 0] if the target is arranged to the center point suddenly.
            if (np.sqrt(data['joystick'][idx][0] ** 2 + data['joystick'][idx][1] ** 2)) < 100:
                tmp['polar'].append([np.sqrt(data['joystick'][idx][0] ** 2 + data['joystick'][idx][1] ** 2),
                                    cal_theta(data['joystick'], idx)])
            else:
                tmp['polar'].append([0, 0])
        
        # Standardize to a human's action.
        tmp['polar'] = np.array(tmp['polar'])
        r = tmp['polar'][:, 0]
        theta = tmp['polar'][:, 1]
        rad = np.deg2rad(theta)

        target_data = np.concatenate([r.reshape(-1, 1), rad.reshape(-1, 1)], axis=1)

        # Radian distribution
        plt.rc('font', size=8)
        plt.xlim(0, 6)
        plt.hist(r)
        plt.title('Target_r')
        plt.savefig(Path(save_path,'RADIAN.jpg'), dpi=300)
        plt.close('all')

        # Theta distribution
        plt.rc('font', size=8)
        plt.xlim(0, 361)
        plt.hist(theta)
        plt.title('Target_theta')
        plt.savefig(Path(save_path, 'THETA.jpg'), dpi=300)
        plt.close('all')

        # Policy distribution
        plt.rc('font', size=8)
        rbins = np.linspace(0, 6, 13)
        abins = np.linspace(0, 2 * np.pi, 37)

        hist, _, _ = np.histogram2d(rad, r, bins=(abins, rbins))
        A, R = np.meshgrid(abins, rbins)
        self.corr_dict['Target'] = hist.T.reshape(-1, 1)

        fig, ax = plt.subplots(subplot_kw=dict(projection='polar'), figsize=(8, 8))
        normalize = matplotlib.colors.LogNorm()

        pc = ax.pcolormesh(A, R, hist.T, cmap='Oranges', norm=normalize)
        fig.colorbar(pc)
        ax.grid(True)
        plt.savefig(Path(save_path, 'Policy.jpg'), dpi=300)
        plt.close('all')

        print("=" * 50)
        print("The Target's Policy histogram has been saved!")
        print("=" * 50)
        
        return True # r, theta, hist

    def DRL_policy(self):
        """
        Description:
            강화학습 행동 데이터로부터 radian 분포, theta 분포, 그리고 행동정책 분포를 시각화하기 위한 함수

        """
        for path in glob(str(Path(self.root_dir, 'Result', 'DRL_model', '*'))):
            name = path.split('/')[-2]
            data = np.load(Path(path, 'Action.npy'))

            plt.rc('font', size=8)
            RL_action = data

            # RADIAN : [0, 1] -> [0, 6] | THETA : [-1, 1] -> [-180, 180]
            RL_action[:, 0] *= 6
            RL_action[:, 1] *= 180

            DRL_action = []
            for idx in range(RL_action[:, 1].shape[0]):
                if RL_action[:, 1][idx] < 0:
                    DRL_action.append(RL_action[:, 1][idx] + 360)
                else:
                    DRL_action.append(RL_action[:, 1][idx])
            DRL_action[:, 1] = np.array(DRL_action)

            save_path = Path(self.save_dir, 'DRL_model', name)
            save_path.mkdir(parents=True ,exist_ok=True)

            # Radian distribution
            plt.hist(DRL_action[:, 0])
            plt.title('{}_r'.format(name))
            plt.savefig(Path(save_path, 'RADIAN.jpg'), dpi=300)

            # Theta distribution
            plt.hist(DRL_action[:, 1])
            plt.title('{}_theta'.format(name))
            plt.savefig(Path(save_path, 'THETA.jpg'), dpi=300)

            # Policy distribution
            r = RL_action[:, 0]
            theta = RL_action[:, 1]
            rad = np.deg2rad(theta)

            rbins = np.linspace(0, 6, 13)
            abins = np.linspace(0, 2 * np.pi, 37)

            hist, _, _ = np.histogram2d(rad, r, bins=(abins, rbins))
            A, R = np.meshgrid(abins, rbins)
            fig, ax = plt.subplots(subplot_kw=dict(projection='polar'), figsize=(8, 8))
            normalize = matplotlib.colors.LogNorm()
            pc = ax.pcolormesh(A, R, hist.T, cmap='Oranges', norm=normalize)
            cb = fig.colorbar(pc)
            ax.grid(True)
            plt.savefig(Path(save_path, 'Policy.jpg'), dpi=300)

            self.corr_dict[name] = hist.T.reshape(-1, 1)

        print("=" * 50)
        print("The DRL's Policy histogram has been saved!")
        print("=" * 50)

        return True # r, theta, hist

    def corr_matrix(self):
        """
        Description:
            행동정책 분포 간의 Correlation을 Matrix로 시각화하기 위한 함수.

        """
        save_path = Path(self.save_dir, self.subj_name)
        save_path.mkdir(parents=True ,exist_ok=True)

        plt.rc('font', size=8)
        self.corr_dict = dict(sorted(self.corr_dict.items()))
        for name in self.corr_dict.keys():
            for name_ in self.corr_dict.keys():
                corr_list.append(np.corrcoef(self.corr_dict[name].reshape(-1), self.corr_dict[name_].reshape(-1))[0][1])
        corr_list = np.array(corr_list).reshape(-1, len(list(self.corr_dict.keys())))

        plt.rc('font', size=8)
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111)
        cax = ax.matshow(corr_list, cmap='Oranges')
        fig.colorbar(cax)

        for (i, j), z in np.ndenumerate(corr_list):
            ax.text(j, i, '{:0.2f}'.format(z), ha='center', va='center')

        ax.set_xticklabels([''] + list(self.corr_dict.keys()))
        ax.set_yticklabels([''] + list(self.corr_dict.keys()))
        plt.xticks(rotation=30)
        plt.savefig(Path(save_path, 'Correlation.jpg'), dpi=300)

        return True


