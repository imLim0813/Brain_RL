import numpy as np

def cal_theta(data, idx):
    """
    Description:
        데카르트 좌표계 (x, y)를 이용하여 각도 (theta)를 구하는 함수.
    
    Args:
        data : 조이스틱 (x, y)값
        idx : timestep 인덱스

    Returns:
        벡터의 각도
        
    """

    if data[idx][1] == 0 or data[idx][0] == 0:
        return 0

    if np.rad2deg(np.arctan(data[idx][1] / data[idx][0])) > 0:
        # quadrant i
        if data[idx][0] > 0:
            theta = np.rad2deg(np.arctan(data[idx][1] / data[idx][0]))
        # quadrant ii
        elif data[idx][0] < 0:
            theta = np.rad2deg(np.arctan(data[idx][1] / data[idx][0])) + 180

    elif np.rad2deg(np.arctan(data[idx][1] / data[idx][0])) < 0:
        # quadrant iv
        if data[idx][0] > 0:
            theta = 360 + np.rad2deg(np.arctan(data[idx][1] / data[idx][0]))
        # quadrant iii
        elif data[idx][0] < 0:
            theta = np.rad2deg(np.arctan(data[idx][1] / data[idx][0])) + 180

    return theta


def euclidean_distance(state):
    """
    Description:
        커서와 타겟 간의 유클리디안 거리를 구하는 함수
    
    Args:
        state : [커서 x, 커서 y, 타겟 x, 타겟 y]

    Return:
        타겟과 커서 간의 유클리디안 거리

    """
    return np.sqrt((state[2] - state[0]) ** 2 + (state[3] - state[1]) ** 2)