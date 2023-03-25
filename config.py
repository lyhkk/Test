# 模型参数设置

def get_config(args):
    # fill in the config function with the parameters you want to use
    opt = {'train': {   'batch_size': 1, # 16
                        'lr': 4e-6, # 1e-4,
                        'lr_decay': 0.5,
                        'lr_decay_epoch': 20, # 20
                        'num_epochs': 4, # 32
<<<<<<< HEAD
                        'save_freq': 2, # Saving models after ~ epochs
=======
                        'save_freq': 10, # Saving models after ~ epochs
>>>>>>> 98ac2c243d71823ab654d7d0746677816f58f0c6
                        'lambda_L': 0.1,
                        'patience': 10,
                        'upscale_factor': 4,
                        'pretrained': False,
                        'Shuffle': False, # True
                        'is_training': True,
                        'train_dataset': 'Videos/VR-super-resolution/train/HP',
                        'best_model_save_folder': 'model/final_model/my_final-726dual',
                        'save_folder': 'model/final_model/my-726dual', # Common save folder
                        'log_dir': 'train_log', # tensorboard log
                        'model_pth': 'vrcnn_final_epoch_272.pth', # lr change flag
                    },
            'val':  {   'val_dataset_lr': 'Videos/VR-super-resolution/test/VR/LR',
                        'val_dataset_hr': 'Videos/VR-super-resolution/test/VR/GT',
                        'val_dataset': 'Videos/VR-super-resolution/val/eval',
                    },
            'test': {   'batch_size': 1,
                        'test_dataset_lr': 'Videos/VR-super-resolution/test/VR/LR',
                        'test_dataset_hr': 'Videos/VR-super-resolution/test/VR/GT',
                        'pre_result':'./results', # model prediction result
                        'exp_name': '726dual', # experiment name
                    },
            'System': { 'gpus': 1, # 4
                        'num_workers': 4,
                        'cuda': True,
                        'seed': 123,
            }
    }
    return opt