synthesis.py

def synthesis(picture1,picture2):
  #ライブラリインポート
  import cv2
  import numpy as np
  #画像の読み込み
  pic1=cv2.imread('picture1',cv2.IMREAD_COLOR)
  pic2=cv2.imread('picture2',cv2.IMREAD_COLOR)
  #二値化処理
  pic2gray=cv2.imread('picture2',cv2.IMREAD_GRAYSCALE)
  ret, thresh = cv2.threshold(pic2gray, 5, 255, cv2.THRESH_BINARY)
  #輪郭抽出
  contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)
  max_cnt =max(contours, key=lambda x: cv2.contourArea(x))
  #マスク画像の作成
  pic2thresh = cv2.drawContours(pic2gray, [max_cnt], -1, 255, -1)
  cv2.imwrite('pic2thresh2.jpg',np.array(pic2thresh))
  #画像合成前処理
  pic2[pic2thresh<255]=[0,0,0]
  pic1[pic2thresh==255]=[0,0,0]
  cv2.imwrite('pic2thres3.jpg',np.array(pic1))
  #画像合成
  pic3=cv2.add(pic1,pic2)
  cv2.imwrite('add.jpg',np.array(pic3))

