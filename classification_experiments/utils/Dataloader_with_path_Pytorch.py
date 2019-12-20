import torch
from torchvision import datasets
# from dataset.Dataloader_with_path import ImageFolderWithPaths






class ImageFolderWithPaths(datasets.ImageFolder):
    """Custom dataset that includes image file paths. Extends
    torchvision.datasets.ImageFolder
    """

    # override the __getitem__ method. this is the method that dataloader calls
    def __getitem__(self, index):
        # this is what ImageFolder normally returns 
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        # the image file path
        path = self.imgs[index][0]
        # make a new tuple that includes original and the path
        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path


if __name__=="__main__":

    root_path = "/work/vajira/data/kvasir_new_23_class/data/split_0"

    test = ImageFolderWithPaths(root_path)
    print(test[0][2].split("/"))