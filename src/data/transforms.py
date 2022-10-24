import numpy as np
from skimage.transform import resize

class SubsamplePointcloud(object):
    def __init__(self, N):
        self.N = N

    def __call__(self, data):
        data_out = data.copy()
        points = data[None]
        normals = data['normals']

        indices = np.random.randint(points.shape[1], size=self.N)
        data_out[None] = points[:, indices]
        data_out['normals'] = normals[:, indices]

        return data_out

class ResizeImage(object):
    def __init__(self, size, order=1):
        self.size = size
        self.order = order

    def __call__(self, img):
        img_out = resize(img, self.size, order=self.order,
                         clip=False, mode='constant',
                         anti_aliasing=False)
        img_out = img_out.astype(img.dtype)
        return img_out
