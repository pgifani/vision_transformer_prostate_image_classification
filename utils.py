import sys
import numpy as np
import torch
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import auc, roc_curve, precision_recall_curve, accuracy_score, recall_score, f1_score, \
    precision_score





def read_data(path_file, k, seed=2, n_class=2):
    file = open(path_file)
    path = file.readlines()
    train_path = []
    train_label = []
    val_path = []
    val_label = []
    NoCancer = []
    Cancer = []

    if n_class == 2:
        for i in path:
            p = i.split('\n')[0]
            tem = p.split('/')
            if tem[4] == 'NoCancer':
                NoCancer.append(p)
            elif tem[4] == 'Cancer':
                Cancer.append(p)



    NoCancer_data = train_test_split(NoCancer, test_size=k, random_state=seed)
    Cancer_data = train_test_split(Cancer, test_size=k, random_state=seed)
    train_path.extend(NoCancer_data[0])
    train_label.extend(list(np.zeros(len(NoCancer_data[0]))))
    train_path.extend(Cancer_data[0])
    train_label.extend(list(np.ones(len(Cancer_data[0]))))
    val_path.extend(NoCancer_data[1])
    val_label.extend(list(np.zeros(len(NoCancer_data[1]))))
    val_path.extend(Cancer_data[1])
    val_label.extend(list(np.ones(len(Cancer_data[1]))))

    return train_path, train_label, val_path, val_label

def train_one_epoch(model, optimizer, data_loader, device, epoch):
    model.train()
    loss_function = torch.nn.CrossEntropyLoss()
    accu_loss = torch.zeros(1).to(device)
    accu_num = torch.zeros(1).to(device)
    optimizer.zero_grad()

    sample_num = 0
    data_loader = tqdm(data_loader)
    for step, data in enumerate(data_loader):
        images, labels = data
        sample_num += images.shape[0]

        pred = model(images.to(device))
        pred_classes = torch.max(pred, dim=1)[1]
        accu_num += torch.eq(pred_classes, labels.to(device)).sum()
        labels = labels.long()
        loss = loss_function(pred, labels.to(device))
        loss.backward()
        accu_loss += loss.detach()

        data_loader.desc = "[train epoch {}] loss: {:.3f}, acc: {:.3f}".format(epoch,
                                                                               accu_loss.item() / (step + 1),
                                                                               accu_num.item() / sample_num)

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        optimizer.step()
        optimizer.zero_grad()

    return accu_loss.item() / (step + 1), accu_num.item() / sample_num


@torch.no_grad()
def evaluate(model, data_loader, device, epoch):
    loss_function = torch.nn.CrossEntropyLoss()

    model.eval()

    accu_num = torch.zeros(1).to(device)  
    accu_loss = torch.zeros(1).to(device)  

    sample_num = 0
    data_loader = tqdm(data_loader)
    for step, data in enumerate(data_loader):
        images, labels = data
        sample_num += images.shape[0]

        pred = model(images.to(device))
        pred_classes = torch.max(pred, dim=1)[1]
        accu_num += torch.eq(pred_classes, labels.to(device)).sum()
        labels = labels.long()
        loss = loss_function(pred, labels.to(device))
        accu_loss += loss

        data_loader.desc = "[valid epoch {}] loss: {:.3f}, acc: {:.3f}".format(epoch,
                                                                               accu_loss.item() / (step + 1),
                                                                               accu_num.item() / sample_num)

    return accu_loss.item() / (step + 1), accu_num.item() / sample_num


@torch.no_grad()
def cal_m(model, data_loader, device, alo='def', num=2):
    model.eval()
    sample_num = 0
    label = np.array([])
    score = np.array([])
    pre_label = np.array([])
    data_loader = tqdm(data_loader)

    for step, data in enumerate(data_loader):
        images, labels = data
        sample_num += images.shape[0]
        pred = model(images.to(device))
        pred_classes = torch.max(pred, dim=1)[1]
        pre_label_t = pred_classes.cpu().numpy()
        if num == 2:
            score_t = torch.softmax(pred, 1).cpu().numpy()
            score_t = score_t[:, 1]
            label = np.append(label, labels.numpy())
            score = np.append(score, score_t)
            pre_label = np.append(pre_label, pre_label_t)
        else:
            score_t = torch.softmax(pred, 1).cpu().numpy()
            label = np.append(label, labels.numpy())
            score = np.append(score, score_t)
            pre_label = np.append(pre_label, pre_label_t)

    if num == 2:
        fpr, tpr, threshold = roc_curve(label, score)
        pre, rec_, _ = precision_recall_curve(label, score)
        acc = accuracy_score(label, pre_label)
        rec = recall_score(label, pre_label)
        f1 = f1_score(label, pre_label)
        Pre = precision_score(label, pre_label)
        au = auc(fpr, tpr)
        apr = auc(rec_, pre)
        f = open("./res_dir/res.txt", 'a')
        f.write(alo + '\n')
        f.write(str(round(Pre, 4)) + '\t')
        f.write(str(round(rec, 4)) + '\t')
        f.write(str(round(acc, 4)) + '\t')
        f.write(str(round(f1, 4)) + '\t')
        f.write(str(round(au, 4)) + '\t')
        f.write(str(round(apr, 4)) + '\n\n')
        f.close()
        print('Precision is :{}'.format(Pre))
        print('Recall is :{}'.format(rec))
        print("ACC is: {}".format(acc))
        print("F1 is: {}".format(f1))
        print("AUC is: {}".format(au))
        print('AUPR is :{}'.format(apr))
    else:
        acc = accuracy_score(label, pre_label)
        print(score.shape)
        score = score.reshape((-1, 3))
        
        f = open("./res_dir/res.txt", 'a')
        f.write(alo + '\n')
        f.write("ACC: " + str(round(acc, 4)) + '\n')
        f.close()
        print("ACC is: {}".format(acc))
    return score
