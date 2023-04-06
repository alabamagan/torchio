import torch
import torch.distributed as dist
from torch.utils.data import DistributedSampler
from .queue import Queue
from .. import NUM_SAMPLES
from .dataset import SubjectsDataset
from .sampler import PatchSampler
from .subject import Subject

from typing import Optional

class QueueDDP(Queue):
    r"""Wrapper for distributed training.

    This class wraps :class:`~torchio.data.Queue` and add two key arguements for DDP.
    Namely the `num_of_replicas`, which often is the world size and also `rank`, which
    dictates the behavior of the loader. Note that the argument `start_background` is disabled.

    Args:
        num_of_replicas:
            The number of queue that will be running parallelly. Default to the value returned
            by :func:`torch.distributed.get_world_size`.
        rank:
            The rank of the queue instance. Default to the value returned by :func:`torch.distributed.
            get_rank`.

    See Also:
        For other initialization arguments, please see: :class:`~torch.data.Queue`

    """ # noqa: E501
    def __init__(self,
                 subjects_dataset  : SubjectsDataset,
                 max_length        : int,
                 samples_per_volume: int,
                 sampler           : PatchSampler,
                 num_workers       : int             = 0,
                 shuffle_subjects  : bool            = True,
                 shuffle_patches   : bool            = True,
                 start_background  : bool            = True,
                 verbose           : bool            = False,
                 num_of_replicas   : Optional[int]   = None,
                 rank              : Optional[int]   = None,
                 ):
        super(QueueDDP, self).__init__(subjects_dataset,
                                       max_length,
                                       samples_per_volume,
                                       sampler,
                                       num_workers,
                                       shuffle_subjects,
                                       shuffle_patches,
                                       False, # start_background doesn't work here
                                       verbose,
                                       )
        if not dist.is_initialized():
            msg = "This class should always be used for distributed training. However, " \
                  "DDP doesn't seem to have been activated."
            raise RuntimeError(msg)

        self.num_of_replicas = num_of_replicas or dist.get_world_size()
        self.rank = rank or dist.get_rank()

        # separate original
        self.subjects_dataset = SubjectsDataset(
            subjects_dataset._subjects[self.rank:len(subjects_dataset):self.num_of_replicas],
            transform = subjects_dataset._transform
        )
