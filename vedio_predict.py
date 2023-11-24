import os.path as osp
from tqdm.auto import tqdm
import cv2
import os
import torch
import torchfcn
from torch.autograd import Variable
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import PIL
from PIL import Image

#加载模型：
model = torchfcn.models.FCN32s(n_class=21)
checkpoint = torch.load('E:\semantic_segmentation\VOC\logs\\20231122_013859.113221\model_best.pth.tar')
model.load_state_dict(checkpoint['model_state_dict'])
if torch.cuda.is_available():
    model.cuda()

# print(model)
def transform(frame):
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    img = np.array(img, dtype=np.uint8)
    img = img[:, :, ::-1]  # RGB -> BGR
    img = img.astype(np.float64)
    img -= np.array([104.00698793, 116.66876762, 122.67891434])
    img = [img.transpose(2, 0, 1).tolist()]
    img = torch.tensor(img).float()
    return img

def schema(selection):

    cv2.namedWindow("video", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("video", 430, 600)
    cap = cv2.VideoCapture(selection)
    count = 0

    while True:

        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)

        if not ret:
            break
        torch_img = transform(frame)
        torch_img = Variable(torch_img).cuda()
        score = model(torch_img)
        if count < 5:
            print(score)
            count += 1
        lbl_pred = score.detach().max(1)[1].cpu().numpy()[:, :, :]  #转化分类
        colored_pred = cv2.applyColorMap((lbl_pred.squeeze() * 10).astype(np.uint8), cv2.COLORMAP_VIRIDIS) #按照不同类别显示不同颜色区域

        cv2.imshow("video", colored_pred+np.clip(frame - 50, 0, 255).astype(np.uint8))

        key = cv2.waitKey(20)

        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

schema(0)


