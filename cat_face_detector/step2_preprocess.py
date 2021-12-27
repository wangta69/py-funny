import random
import dlib, cv2, os
import pandas as pd
import numpy as np

img_size = 224

def resize_img(im):
    """
    이미지를 정사각형으로 만들어주고 사이즈를 줄인다.
    :param im:
    :return:
    """
    old_size = im.shape[:2] # old_size is in (height, width) format
    ratio = float(img_size) / max(old_size)
    new_size = tuple([int(x*ratio) for x in old_size])
    # new_size should be in (width, height) format
    im = cv2.resize(im, (new_size[1], new_size[0]))
    delta_w = img_size - new_size[1]
    delta_h = img_size - new_size[0]
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)
    new_im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT,
            value=[0, 0, 0])
    return new_im, ratio, top, left

def createModes(dirname):

    # dirname = 'CAT_00'
    base_path = 'assets/data_cat_images/archive/%s' % dirname
    file_list = sorted(os.listdir(base_path))
    random.shuffle(file_list)

    dataset = {
        'imgs': [],
        'lmks': [],
        'bbs': []
    }

    for f in file_list:
        if '.cat' not in f:
            continue

        # read landmarks
        pd_frame = pd.read_csv(os.path.join(base_path, f), sep=' ', header=None)
        # landmarks = (pd_frame.as_matrix()[0][1:-1]).reshape((-1, 2))
        landmarks = (pd_frame.to_numpy()[0][1:-1]).reshape((-1, 2))

        # load image
        img_filename, ext = os.path.splitext(f)

        img = cv2.imread(os.path.join(base_path, img_filename))

        # resize image and relocate landmarks
        img, ratio, top, left = resize_img(img)
        landmarks = ((landmarks * ratio) + np.array([left, top])).astype(np.int)    # landmark의 위치도 이미지 리사이즈에 따라 새로 계산한다.
        bb = np.array([np.min(landmarks, axis=0), np.max(landmarks, axis=0)])    # bound box : 얼굴의 영역을 지정

        dataset['imgs'].append(img)
        dataset['lmks'].append(landmarks.flatten())
        dataset['bbs'].append(bb.flatten())

    np.save('assets/dataset/%s.npy' % dirname, np.array(dataset))

def doLoop():
    dirnames = ['CAT_00', 'CAT_01', 'CAT_02', 'CAT_03', 'CAT_04', 'CAT_05', 'CAT_06']
    for dirname in dirnames:
        createModes(dirname)

doLoop()

