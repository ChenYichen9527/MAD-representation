import os
import os.path
import torch
import random
from configs.parse import loadYaml
from dataloader.dataset_prophesee import PropDataset
from torch.optim import *
import numpy as np
from tensorboardX import SummaryWriter
import tqdm
from flow_models.model import FireFlowNet, EVFlowNet
from loss.flow_loss import EventWarping
from utils.utils_save import creat_model_save_path, save_model



def int_parm():
    seed = 20
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":

    configs_path = r'configs/train_flow_configs.yaml'
    config = loadYaml(configs_path)

    int_parm()
    folder_name, log_file,cog_files = creat_model_save_path(r'./results/train')

    device = 'cuda:0'

    loss_function = EventWarping(config, device)

    flow_model = EVFlowNet(config["model_flow"].copy(), 5).to(device)
    flow_model.train()

    dataset = PropDataset(config, mode='test')
    num_samples = int(len(dataset) / 60)
    train_sampler = torch.utils.data.sampler.RandomSampler(list(range(len(dataset))),num_samples=num_samples,replacement=True)
    loader = torch.utils.data.DataLoader(dataset, batch_size=config['train']['batch_size'],
                                         collate_fn=dataset.custom_collate, sampler=train_sampler)



    # optimizers
    optimizer = Adam(flow_model.parameters(), lr=config['train']['lr'])
    optimizer.zero_grad()

    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.1, total_iters=config['train']['epoch'])

    writer = SummaryWriter(log_file)

    loss = 0
    loss_cm = 0
    loss_smoth = 0
    best_loss = 1e7
    for epoch in range(config['train']['epoch']):
        pbar = tqdm.tqdm(total=len(loader), unit="Batch", unit_scale=True,
                         desc="Epoch: {}".format(epoch),position=0,leave=True)
        train_loss = 0
        train_loss_cm = 0
        train_loss_smoth = 0
        for n, inputs in enumerate(loader):
            # forward
            x = flow_model((inputs["inp_voxel"]).to(device), (inputs["inp_cnt"]).to(device))

            loss = loss_function(x["flow"], inputs["inp_list"].to(device),
                                                      inputs["inp_pol_mask"].to(device))
            writer.add_scalar('train_loss', train_loss / (n + 1), epoch*num_samples+n)

            with torch.no_grad():
                train_loss += loss.item()
                pbar.set_postfix(loss_value=loss.item(), Total_Loss=train_loss / (n + 1),flow_max=x["flow"][0].max())
                pbar.update(1)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        if (train_loss / (n + 1)) < best_loss:
            save_model(folder_name, flow_model)
            best_loss=(train_loss / (n + 1))
        scheduler.step()
