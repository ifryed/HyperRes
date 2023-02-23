import os
import torch
import torch.utils.data as data


class VisionDataset(data.Dataset):
    _repr_indent = 4

    def __init__(self, root, transforms=None, transform_common=None, transform_parallel=None, target_transform=None):
        if isinstance(root, torch._six.string_classes):
            root = os.path.expanduser(root)
        self.root = root

        has_transforms = transforms is not None
        has_separate_transform = (transform_common is not None or transform_parallel is not None) or target_transform is not None
        if has_transforms and has_separate_transform:
            raise ValueError("Only transforms or transform/target_transform can "
                             "be passed as argument")

        # for backwards-compatibility
        self.transform_common = transform_common
        self.transform_parallel = transform_parallel
        self.target_transform = target_transform

        if has_separate_transform:
            transforms = StandardTransform(transform_common, transform_parallel, target_transform)
        self.transforms = transforms

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def __repr__(self):
        head = "Dataset " + self.__class__.__name__
        body = ["Number of datapoints: {}".format(self.__len__())]
        if self.root is not None:
            body.append("Root location: {}".format(self.root))
        body += self.extra_repr().splitlines()
        if hasattr(self, "transforms") and self.transforms is not None:
            body += [repr(self.transforms)]
        lines = [head] + [" " * self._repr_indent + line for line in body]
        return '\n'.join(lines)

    def _format_transform_repr(self, transform, head):
        lines = transform.__repr__().splitlines()
        return (["{}{}".format(head, lines[0])] +
                ["{}{}".format(" " * len(head), line) for line in lines[1:]])

    def extra_repr(self):
        return ""


class StandardTransform(object):
    def __init__(self, transform_common=None, transform_parallel=None, target_transform=None):
        self.transform_common = transform_common
        self.transform_parallel = transform_parallel
        self.target_transform = target_transform

    def __call__(self, input, target):
        if self.transform_common is not None:
            input = self.transform_common(input)
        if self.transform_parallel is not None:
            assert isinstance(self.transform_parallel, list)
            transformed_input = self.transform_parallel(input)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return input, target

    def _format_transform_repr(self, transform, head):
        lines = transform.__repr__().splitlines()
        return (["{}{}".format(head, lines[0])] +
                ["{}{}".format(" " * len(head), line) for line in lines[1:]])

    def __repr__(self):
        body = [self.__class__.__name__]
        if self.transform_common is not None:
            body += self._format_transform_repr(self.transform,
                                                "Transform: ")
        if self.transform_parallel is not None:
            for transform in transform_parallel:
                body += self._format_transform_repr(transform,
                                                "Transform: ")
        if self.target_transform is not None:
            body += self._format_transform_repr(self.target_transform,
                                                "Target transform: ")

        return '\n'.join(body)
