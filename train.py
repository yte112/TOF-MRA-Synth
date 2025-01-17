from model.plmodel import MyPlModel, SaveCheck
from utils.dataset import MriForGANSingle, dict_as_namespace, add_minus
from torch.utils.data import DataLoader
import pytorch_lightning as pl

options = {
    'pro_name': 'TOF_Synth',
    # TOF-MRA synth add input gate for clinical, Single dataset 2024/12/26
    'use_modality': ['3D-T1W', '3D-T2W', '3D-FLAIR'], 
    'data_root': 'hdf5-root',
    'epochs': 200,
    'batch_size': 16,
    'lr': 0.002, 
    'lr_scheduler_stages': 6,
    'use_slice': 5,
    'save_freq': 1,
    'start': 0,
    'cuda_num': [4, 5, 6],
    'loss': {
        'temp': 0.5,
        'lamda': 1000
    }
}
options = dict_as_namespace(options)


def train(options):
    # load dataset
    root = options.data_root
    train_dataset = MriForGANSingle('train', options.use_slice, root, options.use_modality)
    val_dataset = MriForGANSingle('valid', options.use_slice, root, options.use_modality)
    train_dataloader = DataLoader(train_dataset, batch_size=options.batch_size, shuffle=True, num_workers=15)
    val_dataloader = DataLoader(val_dataset, batch_size=4, shuffle=True, num_workers=15)
    l_train = len(train_dataloader)
    l_val = len(val_dataloader)
    SaveCalls = [SaveCheck(options)]

    # load model
    model_pl = MyPlModel(options, l_train, l_val)
    trainer = pl.Trainer(
        accumulate_grad_batches=1, 
        accelerator='gpu',
        devices=options.cuda_num, 
        max_epochs=options.epochs,
        precision=32,
        callbacks=SaveCalls,
        num_sanity_val_steps=0,
        enable_progress_bar=False,
        logger=False,
        strategy=pl.strategies.ddp.DDPStrategy(find_unused_parameters=True),
    )

    if options.start != 0:
        trainer.fit(
            model=model_pl, 
            train_dataloaders=train_dataloader, 
            val_dataloaders=val_dataloader,
            ckpt_path=f"./logs/checkpoints/epoch_{options.start}.ckpt"
        )
    else:
        trainer.fit(
            model=model_pl, 
            train_dataloaders=train_dataloader, 
            val_dataloaders=val_dataloader,
        )



if __name__ == '__main__':
    train(options)
