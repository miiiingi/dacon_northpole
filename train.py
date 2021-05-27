from torch.utils.data import DataLoader
from part import * 
traindset = dataset_train()
testset = dataset_test()
trainloader = DataLoader(traindset, batch_size=2)
for image, target in trainloader : 
    print(image.shape, target.shape)
    exit()