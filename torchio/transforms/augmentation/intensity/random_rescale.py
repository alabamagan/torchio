from collections import defaultdict
from typing import Tuple, Dict, Union, Optional

import torch
import numpy as np

from ....utils import to_tuple
from ....typing import TypeRangeFloat
from ....data.subject import Subject
from ... import IntensityTransform
from .. import RandomTransform


class RandomRescale(RandomTransform, IntensityTransform):
    r"""Randomly change the mean and variance of an image. 

    Args:
        mean_range: 
        std_range:
        masking_method:
        bg_value:
        label:
        **kwargs: See :class:`~torchio.transforms.Transform` for additional
            keyword arguments.

    .. warning :: This function has no protection against overflow yet, so be mindful when choosing the range!

    Example:
        >>> import torchio as tio

    """  # noqa: E501
    def __init__(
            self,
            mean: Union[float, Tuple[float, float]] = (-100, 100),
            std: Union[float, Tuple[float, float]] = (1, 255),
            masking_method: str = 'bg_value',
            bg_value: float = None,
            label: str = None,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.mean_range = self._parse_range(mean, 'mean')
        self.std_range = self._parse_range(std, 'std', min_constraint=1)
        self.masking_method = masking_method
        self.bg_value = bg_value
        self.label = label

    def apply_transform(self, subject: Subject) -> Subject:
        arguments = defaultdict(dict)
        for name, image in self.get_images_dict(subject).items():
            mean, std = self.get_params(self.mean_range, self.std_range)
            arguments['mean'][name] = mean
            arguments['std'][name] = std
            arguments['masking_method'][name] = self.masking_method
            arguments['bg_value'][name] = self.bg_value
            arguments['label'][name] = self.label

        transform = Rescale(**self.add_include_exclude(arguments))
        transformed = transform(subject)
        return transformed

    def get_params(self,
                   mean_range: Tuple[float, float],
                   std_range: Tuple[float, float]) -> float:
        mean = self.sample_uniform(*mean_range).item()
        std = self.sample_uniform(*std_range)
        return mean, std



class Rescale(IntensityTransform):
    r"""Rescale the mean and variance of the input image to the desired one.

    Args:
        
        **kwargs: See :class:`~torchio.transforms.Transform` for additional
            keyword arguments.


    Example:
        >>> import torchio as tio

    """  # noqa: E501
    def __init__(
            self,
            mean: Union[float, Dict[str, float]],
            std: Union[float, Dict[str, float]],
            masking_method: Union[str, Dict[str, str]] = 'bg_value',
            bg_value: Union[float, Dict[str, float]] = 0,
            label: Union[str, Dict[str, str]] = None,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.mean = mean
        self.std = std
        self.masking_method = masking_method
        self.bg_value = bg_value
        self.label = label
        self.invert_transform = False
        self.args_names = 'mean', 'std', 'masking_method', 'bg_value', 'label'

    def apply_transform(self, subject: Subject) -> Subject:
        mean, std, masking_method, bg_value, label = args = self.mean, self.std, self.masking_method, self.bg_value, \
                                                            self.label
        for name, image in self.get_images_dict(subject).items():
            if self.arguments_are_dict():
                mean, std, masking_method, bg_value, label = [arg[name] for arg in args]

            # Invert transform is same as forward transform, just that he mean and std were different.
            cur_mean, cur_std = compute_mean_std(image.data, masking_method, bg_value, label)
            image.set_data(image.data.sub(cur_mean).mul(std).div(cur_std).add(mean))
        return subject

def compute_mean_std(image: torch.FloatTensor,
                     masking_method='bg_value',
                     bg_value=0,
                     label=None):
    available_mask_methods = ('bg_value', 'label', 'corner-pixel', 'min-value')
    if masking_method == 'bg_value':
        _img_flatten = image[image != bg_value].flatten()
    elif masking_method == 'label':
        if label is None:
            message = (
                'Label was not specified or was not found in subject.'
            )
            raise ArithmeticError(message)
        if not isinstance(label, torch.Tensor):
            message = (
                'Label must be a torch Tensor.'
            )
            raise ArithmeticError(message)
        _img_flatten = image[label > 0].flatten()
    elif masking_method == 'corner-pixel':
        _img_flatten = image.flatten()
        _img_flatten = _img_flatten[_img_flatten != _img_flatten[0]]
    elif masking_method == 'min-value':
        _img_flatten = image[image != image.min()].flatten()
    else:
        message = (
            f'Masking method can only be one of: {available_mask_methods}, got'
            f'"{masking_method}" instead.'
        )
        raise ArithmeticError(message)

    _img_flatten = _img_flatten.float()
    mean, std = _img_flatten.mean().item(), _img_flatten.std().item()
    return mean, std