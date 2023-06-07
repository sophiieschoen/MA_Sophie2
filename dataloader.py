from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader

# Create an instance of the ImageFolder dataset
dataset = ImageFolder(root='/mnt/nfs-students/nnUNet_preprocessed/Dataset007_PETfixedmedianheader/', transform=ToTensor())

# Create a DataLoader for the dataset
dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=0)


'''
import torch
from torch.utils.data import DataLoader

# Assuming you have already defined your dataset object
dataset = MyDataset()

# Instantiate the DataLoader with num_workers set to 0
data_loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=0)

# Iterate over the data loader to get batches of data
for batch in data_loader:
    # Process the batch of data
    inputs, labels = batch
    # Perform your operations on the batched data

'''
'''
from nnunet.training.network_training.nnUNetTrainer import nnUNetTrainer

class CustomTrainer(nnUNetTrainer):
    def setup_data(self, data_loader_num_workers=0):
        super().setup_data(data_loader_num_workers)

        # Überschreibe die Anzahl der Worker im DataLoader auf 0
        self.data_loader_num_workers = 0

        # Führe zusätzliche Anpassungen durch, falls erforderlich

# Instanziierung des CustomTrainers
trainer = CustomTrainer(...)
trainer.initialize(notify=False)
trainer.run_training()
'''