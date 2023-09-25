from torchvision.transforms import transforms
import torchvision


def get_augmentation(train = True, do_hflip=False, do_vflip=False):
    augs = [
        transforms.RandomRotation((-5, 5)),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.2),
        transforms.RandomEqualize(p=0.2),
        transforms.RandomGrayscale(p=0.2)
    ]

    if do_hflip:
        augs = [transforms.RandomHorizontalFlip(0.2)] + augs
    if do_vflip:
        augs = [transforms.RandomVerticalFlip(0.2)] + augs

    if not train:
        augs = []

    post_augs = [transforms.ToTensor()]

    full_aug = transforms.Compose([
        transforms.RandomOrder(augs),
        transforms.Compose(post_augs)
    ])

    return full_aug
