import SegmentationDataset as dat


def load_data(inputs_train, targets_train, inputs_val=None, targets_val=None):
    traindata = dat.SegmentationDataSet(inputs_train, targets_train, transform=True, test=False)

    if inputs_val is not None and targets_val is not None:
        valdata = dat.SegmentationDataSet(inputs_val, targets_val)
        return traindata, valdata

    else:
        return traindata
