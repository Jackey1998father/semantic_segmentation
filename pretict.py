import os.path as osp
from tqdm.auto import tqdm
import cv2
import os
import torch
import torchfcn
from torch.autograd import Variable
import matplotlib.pyplot as plt
from tqdm import tqdm

def predict(model, best_model_path, val_loader, cuda, pic_save_path):
    iteration = 0

    checkpoint = torch.load(best_model_path)
    model.load_state_dict(checkpoint['model_state_dict'])


    if cuda:
        model.cuda()

    n_class = len(val_loader.dataset.class_names)

    filenames = []
    imgsets_file = 'E:\semantic_segmentation\VOC\\benchmark_RELEASE\dataset\\val.txt'
    dataset_dir = 'E:\semantic_segmentation\VOC\\benchmark_RELEASE\dataset'

    for did in open(imgsets_file):
        did = did.strip()
        img_file = osp.join(dataset_dir, 'img/%s.jpg' % did)
        filenames.append(img_file)

    count = 0
    for batch_idx, (data, target) in tqdm(enumerate(val_loader), position=0, leave=True):
        count += 1

        iteration = iteration + 1
        if cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        with torch.no_grad():
            score = model(data)

        # imgs = data.data.cpu()
        lbl_pred = score.data.max(1)[1].cpu().numpy()[:, :, :]   # 转化
        # lbl_true = target.data.cpu()
        if count < 5:

            print("------------预测维度-----------",target.shape)
            print("------------预测数据-----------", target)

        # 遍历每个batch的预测结果
        for i in range(len(lbl_pred)):

            # 显示原始图像
            plt.subplot(1, 2, 1)
            imgs = cv2.imread(filenames[batch_idx])
            imgs = cv2.cvtColor(imgs, cv2.COLOR_BGR2RGB)


            plt.imshow(imgs)
            plt.title('Original Image')

            # 显示预测结果
            plt.subplot(1, 2, 2)
            plt.imshow(lbl_pred[i], cmap="viridis", vmin=0, vmax=n_class-1)
            plt.title('Predicted Segmentation')

            # 保存图像
            save_path = f"prediction_{iteration}.png"
            path = os.path.join(pic_save_path, save_path)
            plt.savefig(path)


    plt.show()

if __name__ == '__main__':


    model = torchfcn.models.FCN32s(n_class=21)
    best_model_path = 'E:\semantic_segmentation\VOC\logs\\20231122_013859.113221\model_best.pth.tar'
    cuda = torch.cuda.is_available()
    kwargs = {'num_workers': 4, 'pin_memory': True} if cuda else {}
    val_loader = torch.utils.data.DataLoader(torchfcn.datasets.SBDClassSeg("E:\semantic_segmentation", split='val', transform=True), batch_size=1, shuffle=False, **kwargs)
    pic_save_path = 'E:\semantic_segmentation\VOC\logs\\20231122_013859.113221\predict_img'
    predict(model, best_model_path, val_loader, cuda, pic_save_path)



