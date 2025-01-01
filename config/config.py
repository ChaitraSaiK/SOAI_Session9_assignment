class Config:
    def __init__(self):
        self.batch_size = 64
        self.name = "resnet_50"
        self.workers = 4
        self.lr = 0.1
        self.momentum = 0.9
        self.weight_decay = 1e-4
        self.lr_step_size = 30
        self.lr_gamma = 0.1
        
        # Data paths
        self.train_folder = 'path/to/train'  # Update with your path
        self.val_folder = 'path/to/val'      # Update with your path
        self.val_labels_file = 'path/to/val_labels.txt'  # Update with your path
        
        # Data preparation settings
        self.prepare_validation = True  # Set to True to organize validation data
        
        # Training settings
        self.resume_training = True
        self.num_epochs = 100
        self.device = "cuda"
        
    def __repr__(self):
        return str(self.__dict__)
    
    def __eq__(self, other):
        return self.__dict__ == other.__dict__ 