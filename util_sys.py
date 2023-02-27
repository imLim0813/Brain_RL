import os

def os_driver(tmp=False):
    """
    Description:
        Pygame 콘솔 창을 출력할 것인지 여부를 결정하기 위한 함수

    """
    if not tmp:
        os.environ["SDL_VIDEODRIVER"] = "dummy"
        os.environ["SDL_AUDIODRIVER"] = "dummy"