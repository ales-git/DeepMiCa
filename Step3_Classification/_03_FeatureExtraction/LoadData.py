from torchvision import transforms
import random
import torchvision.transforms.functional as TF
try:
    import CustomDataset as dat
except:
    from . import CustomDataset as dat

def load_data(name,train_path,train_txt_path,val_path,val_txt_path,test_path=None,test_txt_path=None,extract=False):
    # if extract==True then no augmentation on training set

    class MyRotationTransform:
        """Rotate by one of the given angles."""

        def __init__(self, angles):
            self.angles = angles

        def __call__(self, x):
            angle = random.choice(self.angles)
            return TF.rotate(x, angle)


    dim = 224
        
    transform = transforms.Compose(
        [MyRotationTransform(angles=[-90, 0, 90]),
         transforms.RandomHorizontalFlip(p=0.5),
         transforms.RandomVerticalFlip(p=0.5)
         ])

    if extract == False:
        traindata = dat.CustomDataset(train_path, train_txt_path, dim, transform)
        valdata = dat.CustomDataset(val_path, val_txt_path, dim)

        return traindata, valdata
    else:
        traindata = dat.CustomDataset(train_path, train_txt_path, dim, extract_name=True)
        valdata = dat.CustomDataset(val_path, val_txt_path, dim, extract_name = True)
        testdata = dat.CustomDataset(test_path, test_txt_path, dim, extract_name=True)

        return traindata, valdata, testdata


