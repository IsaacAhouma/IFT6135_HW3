import argparse
import os
import torchvision
import torchvision.transforms as transforms
import torch
import classify_svhn
from classify_svhn import Classifier

import scipy
import numpy as np 
from inception import InceptionV3

SVHN_PATH = "svhn"
PROCESS_BATCH_SIZE = 32


def get_sample_loader(path, batch_size):
    """
    Loads data from `[path]/samples`

    - Ensure that path contains only one directory
      (This is due ot how the ImageFolder dataset loader
       works)
    - Ensure that ALL of your images are 32 x 32.
      The transform in this function will rescale it to
      32 x 32 if this is not the case.

    Returns an iterator over the tensors of the images
    of dimension (batch_size, 3, 32, 32)
    """
    data = torchvision.datasets.ImageFolder(
        path,
        transform=transforms.Compose([
            transforms.Resize((32, 32), interpolation=2),
            classify_svhn.image_transform
        ])
    )
    data_loader = torch.utils.data.DataLoader(
        data,
        batch_size=batch_size,
        num_workers=2,
    )
    return data_loader


def get_test_loader(batch_size):
    """
    Downloads (if it doesn't already exist) SVHN test into
    [pwd]/svhn.

    Returns an iterator over the tensors of the images
    of dimension (batch_size, 3, 32, 32)
    """
    testset = torchvision.datasets.SVHN(
        SVHN_PATH, split='test',
        download=True,
        transform=classify_svhn.image_transform
    )
    testloader = torch.utils.data.DataLoader(
        testset,
        batch_size=batch_size,
    )
    return testloader


def extract_features(classifier, data_loader):
    """
    Iterator of features for each image.
    """
    with torch.no_grad():
        for x, _ in data_loader:
            x = x.to(device)
            # added an adaptive pooling layer to get a scaler output 
            # for each input using avg pooling
            if isinstance(classifier,InceptionV3):
                h = classifier(x)[0]
                h = torch.squeeze(h) 
            else:
                h = classifier.extract_features(x)
            #h = torch.unsqueeze(h,dim=0)
            #adaptive_pool = torch.nn.AdaptiveAvgPool1d(1)
            
            #h = torch.squeeze(adaptive_pool(h)).detach().cpu().numpy()
            h = h.cpu().numpy()
            for i in range(h.shape[0]):
                yield h[i]

def calculate_fid_score(sample_feature_iterator,
                        testset_feature_iterator):

    # Code as explained in GANs Trained by a Two Time-Scale Update Rule
    # Converge to a Local Nash Equilibrium https://arxiv.org/pdf/1706.08500.pdf
    # converting the iterator to list
    sample_features_list = list(sample_feature_iterator)
    testset_feature_list = list(testset_feature_iterator)
    dim = sample_features_list[0].shape[-1]
    sample_features = np.empty((len(sample_features_list),dim))
    testset_features =  np.empty((len(testset_feature_list),dim))

    # filling the feature set 
    feat_indx = 0
    for feat in sample_features_list:
        sample_features[feat_indx] = feat
        feat_indx +=1

    feat_indx = 0
    for feat in testset_feature_list:
        testset_features[feat_indx] = feat
        feat_indx +=1
    
    # calculating the mean of the activations across samples ground truth and predicted samples
    mean_1 = np.mean(sample_features,axis=0)
    mean_2 = np.mean(testset_features ,axis=0)

    # calculating covariance of input features for both ground truth and generated
    covariance_1 = np.cov(sample_features, rowvar=False)
    covariance_2 = np.cov(testset_features, rowvar=False)

    # converting mean scalar to numpy array 
    mean_1 = np.atleast_1d(mean_1)
    mean_2 = np.atleast_1d(mean_2)

    # converting cov to at least 2 dimensional array
    covariance_1 = np.atleast_2d(covariance_1)
    covariance_2 = np.atleast_2d(covariance_2)

    diff = mean_1 - mean_2
    # product might be almost singular
    covmean, _ = scipy.linalg.sqrtm(covariance_1.dot(covariance_2), disp=False)
    # caclulating covmean and sqrt(sigma1 dot sigma2) and checking 
    # if its values are not finite  as we might get a negative under the sqrt  
    if not np.isfinite(covmean).all():
        # so adding an offset should fix this imaginary error
        offset = np.eye(covariance_1.shape[0]) * eps
        covmean = scipy.linalg.sqrtm((covariance_1 + offset).dot(covariance_2 + offset))

    # numerical error might give some imaginary components
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    # calculating the trace
    trace_covmean = np.trace(covmean)
    return diff.dot(diff) + np.trace(covariance_1) + np.trace(covariance_2) - 2 * trace_covmean



if __name__ == "__main__":
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

    parser = argparse.ArgumentParser(
        description='Score a directory of images with the FID score.')
    parser.add_argument('--model', type=str, default="svhn_classifier.pt",
                        help='Path to feature extraction model.')

    parser.add_argument('--use_inception', type=bool,
                    help='Use Inception model', default=False)
    parser.add_argument('directory', type=str,
                        help='Path to image directory')
    args = parser.parse_args()

    quit = False
    if not(args.use_inception):
        if not os.path.isfile(args.model):
            print("Model file " + args.model + " does not exist.")
            quit = True
        if not os.path.isdir(args.directory):
            print("Directory " + args.directory + " does not exist.")
            quit = True
    if quit:
        exit()
    print("Test")
    if args.use_inception:
        print('Loading inception')
        dims = 2048
        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
        classifier = InceptionV3([block_idx]).to(device)
    else:
        if torch.cuda.is_available():
            classifier = torch.load(args.model).to(device)
        else:
            classifier = torch.load(args.model, map_location='cpu')
    classifier.eval()

    sample_loader = get_sample_loader(args.directory,
                                      PROCESS_BATCH_SIZE)
    sample_f = extract_features(classifier, sample_loader)

    test_loader = get_test_loader(PROCESS_BATCH_SIZE)
    test_f = extract_features(classifier, test_loader)

    fid_score = calculate_fid_score(sample_f, test_f)
    print("FID score:", fid_score)