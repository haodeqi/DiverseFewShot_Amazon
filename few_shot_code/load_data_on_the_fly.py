from few_shot_code.DataProcessing import NlcDatasetSingleFile
from torchtext import data

def get_iter_by_task_id_on_the_fly(TEXT, working_dir,trainfile, devfile, testfile, device, batch_size):
    LABEL = data.Field(sequential=False)
    train, dev, test = NlcDatasetSingleFile.splits(
    TEXT, LABEL, path=working_dir, train=trainfile, validation=devfile, test=testfile)
    train_iter, dev_iter, test_iter = data.BucketIterator.splits(
        (train, dev, test), batch_size=batch_size, device=device)
    train_iter.repeat = False
    return train_iter, dev_iter, test_iter