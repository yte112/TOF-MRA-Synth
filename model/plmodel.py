import os
import torch
from tqdm import tqdm
from utils.acc_list import TestImage, sample_images
from model.pix2pix import GeneratorUNet, Discriminator, weights_init_normal
import pytorch_lightning as pl
from utils.loss import MIPloss, MIXloss


class MyPlModel(pl.LightningModule):
    def __init__(self, options, l_train, l_val) -> None:
        super().__init__()
        self.options = options
        self.use_modality = options.use_modality
        self.G = GeneratorUNet(options.use_slice, options.use_slice)
        self.D = Discriminator((len(self.use_modality) + 1)*options.use_slice)
        self.G.apply(weights_init_normal)
        self.D.apply(weights_init_normal)
        self.mse = torch.nn.MSELoss()
        self.mip_loss = MIPloss(options)
        self.save_acc = TestImage(options.start)
        self.l_train = l_train
        self.l_val = l_val
        self.automatic_optimization = False

    def configure_optimizers(self):
        opt_G = torch.optim.Adam(self.G.parameters(), lr=self.options.lr, betas=(0.5, 0.999))
        opt_D = torch.optim.Adam(self.D.parameters(), lr=self.options.lr, betas=(0.5, 0.999))

        # adaptative step size: depends on the total number of epochs
        step_size = max(int(self.options.epochs) // 6, 1)
        scheduler_G = torch.optim.lr_scheduler.StepLR(opt_G, step_size=step_size, gamma=0.5)
        scheduler_D = torch.optim.lr_scheduler.StepLR(opt_D, step_size=step_size, gamma=0.5)

        return [opt_G, opt_D], [scheduler_G, scheduler_D]

    def forward_G(self, batch):
        input = torch.cat([batch[modal] for modal in self.use_modality], 1)
        img_fake = self.G(input)
        return img_fake
    
    def forward_D(self, batch, img_fake):
        input = torch.cat([batch[modal] for modal in self.use_modality] + [img_fake], 1)
        D_img = self.D(input)
        return D_img

    def training_step(self, batch, batch_idx):
        opt_g, opt_d = self.optimizers()
        # lr_g, lr_d = self.lr_schedulers()

        img_fake = self.forward_G(batch)
        D_fake = self.forward_D(batch, img_fake.detach())

        valid = torch.ones_like(D_fake)
        fake = torch.zeros_like(D_fake)

        ##### opti D
        # self.toggle_optimizer(opt_d)
        # lr_d.step()
        D_fake = D_fake
        loss_fake = self.mse(D_fake, fake)

        D_real  = self.forward_D(batch, batch['3D-TOF-MRA'])
        loss_real = self.mse(D_real, valid)

        loss_D = 50 * (loss_fake + loss_real)

        opt_d.zero_grad()
        self.manual_backward(loss_D)
        opt_d.step()
        # self.lr_scheduler_step(lr_d)
        # lr_d.step()
        
        # self.untoggle_optimizer(opt_d)

        #### opti G
        # self.toggle_optimizer(opt_g)
        D_fake_1 = self.forward_D(batch, img_fake)
        loss_D_1 = self.mse(D_fake_1, valid)
        mip_loss = self.mip_loss(img_fake, batch)
        loss_G = mip_loss['loss'] + loss_D_1

        opt_g.zero_grad()
        self.manual_backward(loss_G)
        opt_g.step()
        # self.lr_scheduler_step(lr_g)
        # lr_g.step()        
        
        # self.untoggle_optimizer(opt_g)
        
        self.save_acc.make_list(mip_loss, batch_idx, self.l_train, 'train')


    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            img_fake = self.forward_G(batch)
            loss = self.mip_loss(img_fake, batch)
            self.save_acc.make_list(loss, batch_idx, self.l_val, 'val')
            if batch_idx == 0:
                sample_images(self.current_epoch, batch, self.G, self.use_modality)
    
    

class SaveCheck(pl.Callback):
    def __init__(self, options):
        super().__init__()
        self.save_freq = options.save_freq
        self.start = options.start
        if not os.path.exists(f'./logs/checkpoints'):
            os.makedirs(f'./logs/checkpoints')
        if not os.path.exists(f'./logs/images/'):
            os.makedirs(f'./logs/images/')

    def on_train_epoch_start(self, trainer, pl_module):
        print(f'Epoch: {pl_module.current_epoch + 1}')
        pl_module.G.train()
        pl_module.D.train()
        self.pbar = tqdm(total=pl_module.l_train, ncols=100, unit='batch', colour='red')
    
    def on_train_batch_end(self, *args):
        self.pbar.update(1)
        
    def on_validation_epoch_start(self, trainer, pl_module):
        self.pbar.close()
        pl_module.save_acc.show('train')
        print('val:')
        pl_module.G.eval()

    def on_validation_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch

        if self.start != 0 and self.start == epoch+1:
            pl_module.save_acc.del_more_val()
        pl_module.save_acc.show('val')

        if (epoch+1) % self.save_freq == 0 and epoch != 0:
            trainer.save_checkpoint(f"./logs/checkpoints/epoch_{epoch+1}.ckpt")
            print(f'Save {epoch+1} last Trainer!')

        print('\n')
        pl_module.save_acc.save_results()