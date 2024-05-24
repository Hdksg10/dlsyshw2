import gzip
import struct
from builtins import map
from typing import List, Optional
from ..data_basic import Dataset
import numpy as np


class MNISTDataset(Dataset):
    def __init__(
        self,
        image_filename: str,
        label_filename: str,
        transforms: Optional[List] = None,
    ):
        ### BEGIN YOUR SOLUTION
        self.images, self.labels = MNISTDataset.parse_mnist(image_filename, label_filename)
        self.transforms = transforms
        assert len(self.images) == len(self.labels) 
        self.length = len(self.images)
        # if self.transforms is not None:
        #     for transform in self.transforms:
        #         self.images = list(map(transform, self.images))
        ### END YOUR SOLUTION

    def __getitem__(self, index) -> object:
        ### BEGIN YOUR SOLUTION
        image = self.images[index]
        label = self.labels[index]
        if isinstance(index, slice):
            image = np.reshape(image, (-1, 28, 28, 1))
        else:
            image = np.reshape(image, (28, 28, 1))
        if self.transforms is not None:
            for transform in self.transforms:
                image = transform(image)
        if isinstance(index, slice):
            image = np.reshape(image, (-1, 784,))
        else:
            image = np.reshape(image, (784,))
        return image, label
        # raise NotImplementedError()
        ### END YOUR SOLUTION

    def __len__(self) -> int:
        ### BEGIN YOUR SOLUTION
        return self.length
        # raise NotImplementedError()
        ### END YOUR SOLUTION
        
    def parse_mnist(image_filesname, label_filename):
        """Read an images and labels file in MNIST format.  See this page:
        http://yann.lecun.com/exdb/mnist/ for a description of the file format.

        Args:
            image_filename (str): name of gzipped images file in MNIST format
            label_filename (str): name of gzipped labels file in MNIST format

        Returns:
            Tuple (X,y):
                X (numpy.ndarray[np.float32]): 2D numpy array containing the loaded
                    data.  The dimensionality of the data should be
                    (num_examples x input_dim) where 'input_dim' is the full
                    dimension of the data, e.g., since MNIST images are 28x28, it
                    will be 784.  Values should be of type np.float32, and the data
                    should be normalized to have a minimum value of 0.0 and a
                    maximum value of 1.0.

                y (numpy.ndarray[dypte=np.int8]): 1D numpy array containing the
                    labels of the examples.  Values should be of type np.int8 and
                    for MNIST will contain the values 0-9.
        """
        ### BEGIN YOUR SOLUTION
        images = MNISTDataset.read_idx_file(image_filesname).astype('float32')
        images = (images - images.min()) / (images.max() - images.min())
        labels = MNISTDataset.read_idx_file(label_filename).astype('uint8')
        return images, labels
        ### END YOUR SOLUTION
        
    def read_idx_file(filename):
        with gzip.open(filename, mode='rb') as fileobj:
            data = fileobj.read()

            (zero1, zero2, type_id, dims), data = MNISTDataset.unpack_part('>bbbb', data)
            if zero1 != 0 or zero2 != 0:
                raise Exception("Invalid file format")

            types = {
                int('0x08', base=16): 'B',
                int('0x09', base=16): 'b',
                int('0x0B', base=16): 'h',
                int('0x0C', base=16): 'i',
                int('0x0D', base=16): 'f',
                int('0x0E', base=16): 'd'
            }
            type_code = types[type_id]

            dim_sizes, data = MNISTDataset.unpack_part('>' + ('i' * dims), data)
            num_examples = dim_sizes[0]
            input_dim = int(np.prod(dim_sizes[1:]))

            X, data = MNISTDataset.unpack_part('>' + (type_code * (num_examples * input_dim)), data)
            if data:
                raise Exception("invalid file format")

            new_shape = (num_examples, input_dim) if input_dim > 1 else num_examples
            return np.array(X).reshape(new_shape, order='C')
        
    def unpack_part(fmt, data):
        size = struct.calcsize(fmt)
        return struct.unpack(fmt, data[:size]), data[size:]