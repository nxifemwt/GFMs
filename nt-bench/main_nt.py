import os

import wandb
from train_utils import finetune

if __name__ == "__main__":
    use_wandb = "WANDB_SWEEP_ID" in os.environ

    class Config:
        def __init__(self):
            self.learning_rate = 5e-5
            self.model_type = "mistral_max_pool"
            self.train_type = "random"
            self.lr_scheduler_type = "cosine"
            self.num_train_epochs = 1
            self.batch_size = 32
            self.warmup_steps = 500
            self.weight_decay = 0.01
            self.fold_number = 0
            self.dataset_name = "H3K4me3"
            self.dataset_base_path = "/home/data/evaluation/genomics/"

    config = Config()
    run = None
    if use_wandb:
        run = wandb.init()
        config = wandb.config
        print(config)
        print(wandb.config)

    finetune(config, run)
