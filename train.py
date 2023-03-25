import os
import torch
import torch.nn as nn
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from Network.modules import VRCNN, dual_network
import argparse
from data.data_utils_debugging import TrainsetLoader, ValidationsetLoader, ycbcr2rgb, mse_weight_loss
from sklearn.model_selection import train_test_split
import numpy as np
from PIL import Image
from train_log.model_log import print_network
import time
from tqdm import tqdm
from tensorboardX import SummaryWriter
from torchvision.transforms import ToPILImage, ToTensor
from metrics.ws_ssim import ws_ssim
from metrics.psnr import ws_psnr, ws_psnr2, psnr, calculate_ssim
import random
from torch.optim.lr_scheduler import StepLR
from config import get_config


# Training settings 建立解析对象
parser = argparse.ArgumentParser(description='PyTorch Super Res Training')

args = parser.parse_args()
# 为parser增加属性

opt = get_config(args)
gpus_list = range(0, opt['System']['gpus'])

seed = opt['System']['seed']
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

# 保存模型的位置
model_name = os.path.join(opt['train']['best_model_save_folder'], opt['train']['model_pth'])

# gpu的编号
device_ids = [Id for Id in range(torch.cuda.device_count())]
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 加载训练数据集
# tr_dataset_hr = opt['train']['train_dataset']()
tr_set = TrainsetLoader(opt['train']['train_dataset'], opt['train']['train_dataset'], opt['train']['upscale_factor'], patch_size=32, n_iters=20) # n_iters=2000
val_set = ValidationsetLoader(opt['val']['val_dataset_lr'], opt['val']['val_dataset_hr'])

# 创建数据加载器
print('===> Loading datasets')
train_loader = DataLoader(tr_set, batch_size=opt['train']['batch_size'], shuffle=opt['train']['Shuffle'], num_workers=opt['System']['num_workers'])
valLoader = DataLoader(dataset=val_set, batch_size=opt['val']['batch_size'], shuffle=False, num_workers=opt['System']['num_workers'])


class Net(nn.Module):
    def __init__(self, vrcnn, dual_net):
        super(Net, self).__init__()
        self.vrcnn = vrcnn
        self.dual_net = dual_net
        
    def forward(self, x):
        output = self.vrcnn(x)
        dual_out = self.dual_net(output)
        return output, dual_out

    
# 模型
print('===> Building model')
vrcnn = VRCNN(opt['train']['upscale_factor'], is_training=opt['train']['is_training'])
dual_net = dual_network()
model = Net(vrcnn, dual_net)
model = model.to(device)
''' 
# DataParallel将模型复制到多个GPU上, 然后将它们的输出合并在一起。这样加快了时间，但是需要更多的显存。 
if torch.cuda.device_count() > 1:
    model = torch.nn.DataParallel(model, device_ids=device_ids) # device_ids = gpus_list
 '''
print_network(model)

# 定义损失函数和优化器
criterion = mse_weight_loss()
criterion = criterion.cuda(device_ids[0])
optimizer = optim.Adam(model.parameters(), lr=opt['train']['lr'], betas=(0.9, 0.999))
# 设置20epochs后学习率减半
scheduler = StepLR(optimizer, step_size=opt['train']['lr_decay_epoch'], gamma=opt['train']['lr_decay'])

# 创建SummaryWriter实例
writer = SummaryWriter(log_dir='./'+opt['train']['log_dir'])


def train():
    train_bar = tqdm(train_loader)
    
    # 训练循环
    best_psnr = 0.0
    best_ssim = 0.0
    # 迭代的初始下标为1
    # for iteration, batch in enumerate(train_loader, 1):
    for epoch in range(opt['train']['num_epochs']):
        
        psnr_loss = 0.0
        running_loss = 0.0
        patience_counter = 0
        iteration = 1
        
        for data in train_bar:
            # model.train()启动batch normalization和Dropout
            model.train()
            
            # batch[0]与[1]的形式都应是tensor(batch_size,img_size)
            # input, target = Variable(batch[0]), Variable(batch[1], requires_grad=False)
            batch_lr_y, label, hr_texture, lr_texture = data
            batch_lr_y, label = Variable(batch_lr_y).cuda(gpus_list[0]), Variable(label).cuda(gpus_list[0])
            # in_tr, in_val, tar_tr, tar_val = train_test_split(input, target, test_size=0.2, random_state=42)
            optimizer.zero_grad()
            
            output, out_dual = model(batch_lr_y)
            # 计算损失
            out = output.squeeze(dim=2)
            print(output.size())
            primary_loss = criterion(out, label)
            dual_loss = opt['train']['lambda_L'] * criterion(out_dual, batch_lr_y)
            loss = primary_loss + dual_loss
            
            writer.add_scalar('Train/LOSS', loss.item(), epoch*len(train_loader) + iteration)
            
            # psnr_value
            psnr_value = psnr(np.array(torch.squeeze(label).data.cpu())*255, np.array(torch.squeeze(output).data.cpu())*255)
            writer.add_scalar('Train/PSNR', psnr_value, epoch*len(train_loader) + iteration)
            psnr_loss += psnr_value
            '''
            ssim_value = calculate_ssim(np.array(torch.squeeze(label).data.cpu())*255, np.array(torch.squeeze(output).data.cpu())*255)
            writer.add_scalar('Train/SSIM', ssim_value, epoch*len(train_loader) + iteration)
            ssim_loss += ssim_value
            '''
            # 反向传播和优化
            loss.backward()
            optimizer.step()

            # 更新损失
            running_loss += loss.item()
            iteration += 1

        # 计算平均损失
        epoch_loss = running_loss / len(train_loader)
        psnr_loss /= len(train_loader)
        # ssim_loss /= len(train_loader)
        
        # 更新学习率
        scheduler.step()

        print("===> Epoch[{}]({}/{}): Loss: {:.4f} || PSNR_Value: {:.4f}".format(epoch, epoch+1, len(train_loader), epoch_loss, psnr_loss))
        torch.save(model.state_dict(), model_name)
        # writer.close()
        # Validation
        print('===> Start Validation')
        model.eval()
        val_psnr, val_ssim = validate(valLoader, model)
        print('Epoch {}/{} PSNR: {:.6f} SSIM: {:.6f}'.format(epoch, opt['train']['num_epochs'], val_psnr, val_ssim))
        writer.add_scalar('Validation/PSNR', val_psnr, epoch)
        writer.add_scalar('Validation/SSIM', val_ssim, epoch)

        # 保存最好的模型
        if val_psnr > best_psnr and val_ssim > best_ssim:
            best_psnr = val_psnr
            torch.save(model.state_dict(), model_name)
            print('===> Saving Model')
            patience_counter = 0
        else:
            if (epoch + 1) % opt['train']['save_freq'] == 0:
                save_model(model, epoch + 1)
            patience_counter += 1

        # 提前停止
        # writer.close()
        if patience_counter >= opt['train']['patience']:
            print("Early stopping")
            break


def validate(valLoader, model):
    with torch.no_grad():
        ave_psnr = 0
        ave_ssim = 0
        val_bar = tqdm(valLoader)
        for data in val_bar:
            model.eval()
            # dual_net.eval()
            batch_lr_y, label, SR_cb, SR_cr, idx, bicubic_restore = data
            batch_lr_y, label = Variable(batch_lr_y).cuda(gpus_list[0]), Variable(label).cuda(gpus_list[0])
            output, _ = model(batch_lr_y)
            
            SR_ycbcr = np.concatenate((np.array(output.squeeze(0).data.cpu()), SR_cb, SR_cr), axis=0).transpose(1, 2, 0)
            SR_rgb = ycbcr2rgb(SR_ycbcr) * 255.0
            SR_rgb = np.clip(SR_rgb, 0, 255)
            SR_rgb = ToPILImage()(SR_rgb.astype(np.uint8))
            #ToTensor() ---image(0-255)==>image(0-1), (H,W,C)==>(C,H,W)
            SR_rgb = ToTensor()(SR_rgb)

            psnr_value = ws_psnr(np.array(torch.squeeze(label).data.cpu())*255, np.array(torch.squeeze(output).data.cpu())*255)
            ave_psnr = ave_psnr + psnr_value
            ssim_value = ws_ssim(np.array(torch.squeeze(label).data.cpu())*255, np.array(torch.squeeze(output).data.cpu())*255)
            ave_ssim = ave_ssim + ssim_value
    return ave_psnr / len(valLoader), ave_ssim / len(valLoader)


def save_model(model, epoch):
    # 构造保存路径
    save_path = os.path.join(opt['train']['save_folder'], 'model-{}.ckpt'.format(epoch))
    # 保存模型
    torch.save(model.state_dict(), save_path)
    print('Model saved to {}'.format(save_path))

    
if __name__ == '__main__':
    print('===> Start Training')
    train()
    print('===> Training Finished')
    writer.close()