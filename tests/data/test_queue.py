from parameterized import parameterized
import torch
import torchio as tio
from torch.utils.data import DataLoader
from torchio.data import UniformSampler
from torchio.utils import create_dummy_dataset
import os
import torch.distributed as dist
import torch.multiprocessing as mp

from ..utils import TorchioTestCase


class TestQueue(TorchioTestCase):
    """Tests for `queue` module."""

    def setUp(self):
        super().setUp()
        self.subjects_list = create_dummy_dataset(
            num_images=10,
            size_range=(10, 20),
            directory=self.dir,
            suffix='.nii',
            force=False,
        )

    def run_queue(self, num_workers=0, **kwargs):
        subjects_dataset = tio.SubjectsDataset(self.subjects_list)
        patch_size = 10
        sampler = UniformSampler(patch_size)
        queue_dataset = tio.Queue(
            subjects_dataset,
            max_length=6,
            samples_per_volume=2,
            sampler=sampler,
            num_workers=num_workers,
            **kwargs,
        )
        _ = str(queue_dataset)
        batch_loader = DataLoader(queue_dataset, batch_size=4)
        for batch in batch_loader:
            _ = batch['one_modality'][tio.DATA]
            _ = batch['segmentation'][tio.DATA]
        return queue_dataset

    def test_queue(self):
        self.run_queue(num_workers=0)

    def test_queue_multiprocessing(self):
        self.run_queue(num_workers=2)

    def test_queue_no_start_background(self):
        self.run_queue(num_workers=0, start_background=False)

    @parameterized.expand([(11,), (12,)])
    def test_different_samples_per_volume(self, max_length):
        image2 = tio.ScalarImage(tensor=2 * torch.ones(1, 1, 1, 1))
        image10 = tio.ScalarImage(tensor=10 * torch.ones(1, 1, 1, 1))
        subject2 = tio.Subject(im=image2, num_samples=2)
        subject10 = tio.Subject(im=image10, num_samples=10)
        dataset = tio.SubjectsDataset([subject2, subject10])
        patch_size = 1
        sampler = UniformSampler(patch_size)
        queue_dataset = tio.Queue(
            dataset,
            max_length=max_length,
            samples_per_volume=3,  # should be ignored
            sampler=sampler,
        )
        batch_loader = DataLoader(queue_dataset, batch_size=6)
        batches = [batch['im'][tio.DATA] for batch in batch_loader]
        all_numbers = torch.stack(batches).flatten().tolist()
        assert all_numbers.count(10) == 10
        assert all_numbers.count(2) == 2

    def test_get_memory_string(self):
        queue = self.run_queue()
        memory_string = queue.get_max_memory_pretty()
        assert isinstance(memory_string, str)

    def test_queue_order(self):
        for i in range(len(self.subjects_list)):
            self.subjects_list[i]['ID'] = i
        subjects_dataset = tio.SubjectsDataset(self.subjects_list)
        patch_size = 10
        sampler = UniformSampler(patch_size)
        queue_dataset = tio.Queue(
            subjects_dataset,
            max_length=6,
            samples_per_volume=1,
            sampler=sampler,
            num_workers=2,
            shuffle_subjects=False,
            shuffle_patches=False
        )
        batch_loader = DataLoader(queue_dataset, batch_size=3, shuffle=False, num_workers=0, drop_last=False)
        self.assertEqual(list(range(len(queue_dataset))),
                         [x['ID'] for x in subjects_dataset])
        queue_dataset.patches_list.clear()
        queue_dataset._initialize_subjects_iterable()
        sequential_idlist = []
        for i, mb in enumerate(queue_dataset):
            if i == len(subjects_dataset):
                break
            sequential_idlist.append(mb['ID'])
        self.assertEqual(list(range(len(subjects_dataset))),
                         sequential_idlist)
        queue_dataset.patches_list.clear()
        queue_dataset._initialize_subjects_iterable()
        sequential_idlist = []
        for mb in batch_loader:
            sequential_idlist.extend(mb['ID'])
        self.assertEqual(list(range(len(subjects_dataset))),
                         sequential_idlist)


class TestQueueDDP(TorchioTestCase):
    def setUp(self):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("No cuda device to test distributed queue.")

        if torch.cuda.device_count() <= 1:
            raise unittest.SkipTest("At list two cuda devices to test distributed queue.")

        super(TestQueueDDP, self).setUp()
        self.subjects_list = create_dummy_dataset(
            num_images=10,
            size_range=(10, 20),
            directory=self.dir,
            suffix='.nii',
            force=False,
        )

        self.backbone = 'nccl'
        self.world_size = torch.cuda.device_count()

    @staticmethod
    def run_queue(rank, subjects_list, backbone,world_size, num_workers=0):
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '23455'
        dist.init_process_group(backbone, world_size=world_size, rank=rank)
        subjects_dataset = tio.SubjectsDataset(subjects_list)
        patch_size = 10
        sampler = UniformSampler(patch_size)
        queue_dataset = tio.QueueDDP(
            subjects_dataset,
            max_length=6,
            samples_per_volume=2,
            sampler=sampler,
            num_workers=num_workers,
            num_of_replicas=world_size,
            rank=rank
        )
        _ = str(queue_dataset)
        batch_loader = DataLoader(queue_dataset, batch_size=4, drop_last=True)
        for batch in batch_loader:
            _ = batch['one_modality'][tio.DATA]
            _ = batch['segmentation'][tio.DATA]
        dist.destroy_process_group()
        return queue_dataset

    def test_ddp_run(self):
        for i in range(self.world_size):
            mp.spawn(TestQueueDDP.run_queue, args=(self.subjects_list, self.backbone, self.world_size, 4, ), nprocs=self.world_size)
        pass