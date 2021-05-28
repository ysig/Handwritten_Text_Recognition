import copy
import numpy as np
from skimage import io as img_io
from skimage import transform
from skimage import util
from tqdm import tqdm
from params import *

params = BaseOptions().parser()

# ------------------------------------------------
'''
In this block : Define paths to datasets
'''

# PATH TO HDD
# line_gt = '/media/vn_nguyen/hdd/hux/IAM/lines.txt'
# line_img = '/media/vn_nguyen/hdd/hux/IAM/lines/'
# line_train = '/media/vn_nguyen/hdd/hux/IAM/split/trainset.txt'
# line_test = '/media/vn_nguyen/hdd/hux/IAM/split/testset.txt'
# line_val1 = '/media/vn_nguyen/hdd/hux/IAM/split/validationset1.txt'
# line_val2 = '/media/vn_nguyen/hdd/hux/IAM/split/validationset2.txt'


# PATH TO IAM DATASET ON SSD
line_gt = params.tr_data_path + 'IAM/lines.txt'
line_img = params.tr_data_path + 'IAM/lines/'
PP = os.path.dirname(__file__)

# ------------------------------------------------
'''
In this block : Data utils for IAM dataset
'''
def read_lines_text(annotation_txt):
    data = []
    with open(annotation_txt, 'r') as f:
        for line in f.readlines():
            line = line.strip('\n')
            if line.startswith('#'):
                continue
            else:
                spl = line.split(' ')
                image_dir = spl[0].split('-')
                dt = (os.path.join(image_dir[0], image_dir[0]+'-'+image_dir[1], spl[0]+'.png'), ' '.join(spl[8:]))
                data.append(dt)
    return data


class IAM(object):
    def __init__(self, annotation_txt, image_folder):
        self.data = read_lines_text(annotation_txt)
        self.image_folder = image_folder

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        path, txt = self.data[idx]
        return os.path.join(self.image_folder, path), txt
        
    def subset(self, file):
        new_dataset = copy.deepcopy(self)
        valid_key = {str(l.strip('\n')) for l in open(file, 'r').readlines()}
        new_data = []
        for d, i in new_dataset.data:
            ds = os.path.split(os.path.split(d)[0])[1]
            if ds in valid_key:
                new_data.append((d, i))
        new_dataset.data = new_data
        return new_dataset


def gather_iam_line(set='train'):
    '''
    Read given dataset IAM from path line_gt and line_img
    return: List[Tuple(str(image path), str(ground truth))]
    '''
    iam = IAM(line_gt, line_img)
    iam = ()
    if set == 'train':
        dataset = iam.subset(os.path.join(PP, 'splits', 'train.uttlist')),
    elif set in {'val1', 'val2'}:
        dataset = iam.subset(os.path.join(PP, 'splits', 'validation.uttlist')),
    else:
        dataset = iam.subset(os.path.join(PP, 'splits', 'test.uttlist'))

    # gtfile = line_gt
    # root_path = line_img
    # if set == 'train':
    #     data_set = np.loadtxt(line_train, dtype=str)
    # elif set == 'test':
    #     data_set = np.loadtxt(line_test, dtype=str)
    # elif set == 'val':
    #     data_set = np.loadtxt(line_val1, dtype=str)
    # elif set == 'val2':
    #     data_set = np.loadtxt(line_val2, dtype=str)
    # else:
    #     print("Cannot find this dataset. Valid values for set are 'train', 'test', 'val' or 'val2'.")
    #     return
    # gt = []
    # print("Reading IAM dataset...")
    # for line in open(gtfile):
    #     if not line.startswith("#"):
    #         info = line.strip().split()
    #         name = info[0]
    #         name_parts = name.split('-')
    #         pathlist = [root_path] + ['-'.join(name_parts[:i+1]) for i in range(len(name_parts))]
    #         line_name = pathlist[-1]
    #         if (info[1] != 'ok') or (line_name not in data_set):  # if the line is not properly segmented
    #             continue
    #         img_path = '/'.join(pathlist)
    #         transcr = ' '.join(info[8:])
    #         gt.append((img_path, transcr))
    # print("Reading done.")

    return [dataset[i] for i in range(len(dataset))]


def iam_main_loader(set='train'):
    '''
    Store pairs of image and its ground truth text
    return: List[Tuple(nparray(image), str(ground truth text))]
    '''

    line_map = gather_iam_line(set)

    data = []
    for i, (img_path, transcr) in enumerate(tqdm(line_map)):
        try:
            img = img_io.imread(img_path + '.png')
            # img = 1 - img.astype(np.float32) / 255.0
            img = img.astype(np.float32) / 255.0
        except:
            continue
        data += [(img, transcr.replace("|", " "))]

    return data

# ------------------------------------------------


# test the functions
if __name__ == '__main__':
    (img_path, transcr) = gather_iam_line('train')[0]
    img = img_io.imread(img_path + '.png')
    print(img.shape)
    print(img)

    data = iam_main_loader(set='train')
    print("length of train set:", len(data))
    print(data[10][0].shape)

    data = iam_main_loader(set='test')
    print("length of test set:", len(data))

    data = iam_main_loader(set='val')
    print("length of val set:", len(data))
    print("Success")
