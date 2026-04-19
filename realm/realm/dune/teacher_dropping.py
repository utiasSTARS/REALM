import random
from typing import List, Dict

import torch


class TeacherDropping:
    drop_methods = ["none", "lowest_loss", "own_data", "own+generic_data"]

    def __init__(self, method="lowest_loss", p=0.5):
        assert method in self.drop_methods, "Unknown method: {}".format(method)
        assert p == 0 or method == "lowest_loss"
        self.method = method
        self.p = p

    def __call__(self, loss_dict: Dict[str, torch.Tensor], dset_name: List[str]):
        """
        Given a dictionary of losses,
        where keys are teacher names, and values are 2D tensors of shape (B, N)
        (B: batch size and N: number of tokens)
        this function aggregates the losses into a single loss tensor.

        Args:
            loss: (Dict[str, torch.Tensor]
                Loss incurred on each image from each teacher.
            dset_name: (List[str], of shape [B])
                Dataset name for each image
        """
        teachers = sorted(loss_dict.keys())

        loss_tensor = torch.stack([loss_dict[key] for key in teachers])
        B = loss_dict[teachers[0]].shape[0]
        assert loss_tensor.shape == (len(loss_dict), B)
        assert len(dset_name) == B

        if self.method == "none":
            # no drop, all teachers contribute to the loss
            coeffs = torch.ones_like(loss_tensor)

        elif self.method == "lowest_loss":
            # drop teachers with lowest loss,
            # i.e., teacher drop as in UNIC
            coeffs = torch.stack(
                [
                    _get_teacher_coefficients_by_loss(
                        loss_tensor[:, lix], drop_prob=self.p
                    )
                    for lix in range(loss_tensor.shape[1])
                ]
            ).t()

        elif self.method in ["own_data", "own+generic_data"]:
            # drop teachers that do not match the dataset name
            coeffs = torch.zeros_like(loss_tensor)
            for si, dname in enumerate(dset_name):
                tname_si = dataset_to_teacher(dname)
                for ti, tname in enumerate(teachers):
                    if tname.startswith(tname_si) or (
                        self.method == "own+generic_data"
                        and tname_si.startswith("dino2")
                    ):
                        coeffs[ti, si] = 1.0

        else:
            raise NotImplementedError(
                "{}(method={}) is not implemented".format(
                    self.__class__.__name__, self.method
                )
            )

        assert coeffs.shape == loss_tensor.shape

        # make sure each image is assigned to at least one teacher
        assert torch.all(
            coeffs.sum(dim=0) >= 1
        ), "{} images in the batch are not assigned to any of the teachers".format(
            (coeffs.sum(dim=0) == 0).int().sum()
        )

        #####
        # normalize coefficients such that
        # each image contributes to the loss with equal weight
        coeffs.div_(coeffs.sum())
        loss = (coeffs.clone().detach() * loss_tensor).sum()

        # sum the coefficients for each teacher
        # for logging purposes
        coeff_dict = {key: coeff for key, coeff in zip(teachers, coeffs.sum(dim=1))}

        return loss, coeff_dict


@torch.no_grad()
def _get_teacher_coefficients_by_loss(losses, drop_prob=0.5):
    """
    Given a list of losses from all teachers, return a list for their loss coefficients.
    Initially, all coefficients are 1.
    Then we flip coefficients for teachers with lowest loss to zeros with a probability drop_prob.
    """
    if isinstance(losses, (list, tuple)):
        losses = torch.stack(losses)

    # make sure that losses are 1D
    assert len(losses.shape) == 1

    coeffs = torch.ones_like(losses, requires_grad=False)

    # find the teacher with the highest loss
    max_loss_idx = torch.argmax(losses)

    # go through other teachers and
    # flip their coefficients to zeros with a probability drop_prob
    for i in range(len(losses)):
        if i != max_loss_idx:
            p = random.random()
            if p < drop_prob:
                coeffs[i] = 0

    return coeffs
