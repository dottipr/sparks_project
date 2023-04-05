import numpy as np
from torch.nn.modules.loss import _Loss
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from LovaszSoftmax.pytorch import lovasz_losses


################################## FOCAL LOSS ##################################
'''
https://github.com/kornia/kornia
'''
# based on:
# https://github.com/zhezh/focalloss/blob/master/focalloss.py

# add ignore index to focal loss implementation


def one_hot(labels: torch.Tensor,
            num_classes: int,
            device: Optional[torch.device] = None,
            dtype: Optional[torch.dtype] = None,
            eps: float = 1e-6) -> torch.Tensor:
    r"""Converts an integer label x-D tensor to a one-hot (x+1)-D tensor.

    Args:
        labels (torch.Tensor) : tensor with labels of shape :math:`(N, *)`,
                                where N is batch size. Each value is an integer
                                representing correct classification.
        num_classes (int): number of classes in labels.
        device (Optional[torch.device]): the desired device of returned tensor.
         Default: if None, uses the current device for the default tensor type
         (see torch.set_default_tensor_type()). device will be the CPU for CPU
         tensor types and the current CUDA device for CUDA tensor types.
        dtype (Optional[torch.dtype]): the desired data type of returned
         tensor. Default: if None, infers data type from values.

    Returns:
        torch.Tensor: the labels in one hot tensor of shape :math:`(N, C, *)`,

    Examples:
        >>> labels = torch.LongTensor([[[0, 1], [2, 0]]])
        >>> one_hot(labels, num_classes=3)
        tensor([[[[1.0000e+00, 1.0000e-06],
                  [1.0000e-06, 1.0000e+00]],
        <BLANKLINE>
                 [[1.0000e-06, 1.0000e+00],
                  [1.0000e-06, 1.0000e-06]],
        <BLANKLINE>
                 [[1.0000e-06, 1.0000e-06],
                  [1.0000e+00, 1.0000e-06]]]])

    """
    if not isinstance(labels, torch.Tensor):
        raise TypeError("Input labels type is not a torch.Tensor. Got {}"
                        .format(type(labels)))

    if not labels.dtype == torch.int64:
        raise ValueError(
            "labels must be of the same dtype torch.int64. Got: {}" .format(
                labels.dtype))

    if num_classes < 1:
        raise ValueError("The number of classes must be bigger than one."
                         " Got: {}".format(num_classes))

    shape = labels.shape
    one_hot = torch.zeros(
        (shape[0], num_classes) + shape[1:], device=device, dtype=dtype
    )

    return one_hot.scatter_(1, labels.unsqueeze(1), 1.0) + eps


def focal_loss(
        input: torch.Tensor,
        target: torch.Tensor,
        alpha: list = None,
        gamma: float = 2.0,
        reduction: str = 'none',
        ignore_index: int = 9999,
        eps: float = 1e-8,) -> torch.Tensor:
    r"""Criterion that computes Focal loss.

    According to :cite:`lin2018focal`, the Focal loss is computed as follows:

    .. math::

        \text{FL}(p_t) = -\alpha_t (1 - p_t)^{\gamma} \, \text{log}(p_t)

    Where:
       - :math:`p_t` is the model's estimated probability for each class.

    Args:
        input (torch.Tensor): logits tensor with shape :math:`(N, C, *)` where C = number of classes.
        target (torch.Tensor): labels tensor with shape :math:`(N, *)` where each value is :math:`0 ≤ targets[i] ≤ C−1`.
        alpha (float): Weighting factor :math:`\alpha` with shape `(C)`.
        gamma (float, optional): Focusing parameter :math:`\gamma >= 0`. Default 2.
        reduction (str, optional): Specifies the reduction to apply to the
         output: ‘none’ | ‘mean’ | ‘sum’. ‘none’: no reduction will be applied,
         ‘mean’: the sum of the output will be divided by the number of elements
         in the output, ‘sum’: the output will be summed. Default: ‘none’.
        eps (float, optional): Scalar to enforce numerical stabiliy. Default: 1e-8.

    Return:
        torch.Tensor: the computed loss.

    Example:
        >>> N = 5  # num_classes
        >>> input = torch.randn(1, N, 3, 5, requires_grad=True)
        >>> target = torch.empty(1, 3, 5, dtype=torch.long).random_(N)
        >>> output = focal_loss(input, target, alpha=0.5, gamma=2.0, reduction='mean')
        >>> output.backward()
    """

    if not isinstance(input, torch.Tensor):
        raise TypeError("Input type is not a torch.Tensor. Got {}"
                        .format(type(input)))

    if not len(input.shape) >= 2:
        raise ValueError("Invalid input shape, we expect BxCx*. Got: {}"
                         .format(input.shape))

    if input.size(0) != target.size(0):
        raise ValueError('Expected input batch_size ({}) to match target batch_size ({}).'
                         .format(input.size(0), target.size(0)))

    n = input.size(0)
    out_size = (n,) + input.size()[2:]
    if target.size()[1:] != input.size()[2:]:
        raise ValueError('Expected target size {}, got {}'.format(
            out_size, target.size()))

    if not input.device == target.device:
        raise ValueError(
            "input and target must be in the same device. Got: {} and {}" .format(
                input.device, target.device))

    # create mask for ignore_index
    #print("IGNORE INDEX", ignore_index)
    ignore_mask = torch.where(target == ignore_index, 0, 1)
    #print("IGNORE MASK\n", ignore_mask, "\n", ignore_mask.shape)
    target_copy = torch.clone(target).detach()  # do not modify original target
    # convert target to torch.int64
    target_copy = target_copy.to(torch.int64)
    temp_ignored_index = 0  # the corresponding pts will be ignored
    target_copy[target_copy == ignore_index] = temp_ignored_index
    #print("TARGET COPY", target_copy)
    num_ignored = torch.count_nonzero(ignore_mask)
    #print("NUM IGNORED", num_ignored)

    if alpha == None:
        alpha_mask = 1.
    else:
        # create alpha mask that will multiply focal loss
        assert len(
            alpha) == input.shape[1], "alpha does not contain a weight per class"
        alpha_mask = torch.zeros(target.shape)
        for idx, alpha_t in enumerate(alpha):
            alpha_mask[target == idx] = alpha_t
        # print(alpha_mask)

    alpha_mask = alpha_mask.to(target.device)

    # compute softmax over the classes axis
    input_soft: torch.Tensor = F.softmax(input, dim=1) + eps
    #input_soft: torch.Tensor = input + eps

    # create the labels one hot tensor
    target_one_hot: torch.Tensor = one_hot(
        target_copy, num_classes=input.shape[1],
        device=input.device, dtype=input.dtype)
    #print("TARGET ONE HOT", target_one_hot, target_one_hot.shape)

    # compute the actual focal loss
    weight = torch.pow(-input_soft + 1., gamma)
    focal = - weight * torch.log(input_soft)
    #print("ALPHA MASK", alpha_mask)
    #print("FOCAL", focal)
    loss_tmp = torch.sum(target_one_hot * focal, dim=1)
    #print("LOSS TMP SHAPE", loss_tmp.shape)
    # remove loss values in ignored points
    loss_tmp = loss_tmp * ignore_mask * alpha_mask
    #print("LOSS TMP SHAPE", loss_tmp.shape)
    #print("LOSS TMP", loss_tmp)

    if reduction == 'none':
        loss = loss_tmp
    elif reduction == 'mean':
        #loss_old = torch.mean(loss_tmp)
        loss = torch.sum(loss_tmp)/num_ignored
        #print(loss, loss_old)
    elif reduction == 'sum':
        loss = torch.sum(loss_tmp)
    else:
        raise NotImplementedError("Invalid reduction mode: {}"
                                  .format(reduction))
    return loss


''' original version
def focal_loss(
        input: torch.Tensor,
        target: torch.Tensor,
        alpha: float,
        gamma: float = 2.0,
        reduction: str = 'none',
        eps: float = 1e-8) -> torch.Tensor:
    r"""Criterion that computes Focal loss.

    According to :cite:`lin2018focal`, the Focal loss is computed as follows:

    .. math::

        \text{FL}(p_t) = -\alpha_t (1 - p_t)^{\gamma} \, \text{log}(p_t)

    Where:
       - :math:`p_t` is the model's estimated probability for each class.

    Args:
        input (torch.Tensor): logits tensor with shape :math:`(N, C, *)` where C = number of classes.
        target (torch.Tensor): labels tensor with shape :math:`(N, *)` where each value is :math:`0 ≤ targets[i] ≤ C−1`.
        alpha (float): Weighting factor :math:`\alpha \in [0, 1]`.
        gamma (float, optional): Focusing parameter :math:`\gamma >= 0`. Default 2.
        reduction (str, optional): Specifies the reduction to apply to the
         output: ‘none’ | ‘mean’ | ‘sum’. ‘none’: no reduction will be applied,
         ‘mean’: the sum of the output will be divided by the number of elements
         in the output, ‘sum’: the output will be summed. Default: ‘none’.
        eps (float, optional): Scalar to enforce numerical stabiliy. Default: 1e-8.

    Return:
        torch.Tensor: the computed loss.

    Example:
        >>> N = 5  # num_classes
        >>> input = torch.randn(1, N, 3, 5, requires_grad=True)
        >>> target = torch.empty(1, 3, 5, dtype=torch.long).random_(N)
        >>> output = focal_loss(input, target, alpha=0.5, gamma=2.0, reduction='mean')
        >>> output.backward()
    """
    if not isinstance(input, torch.Tensor):
        raise TypeError("Input type is not a torch.Tensor. Got {}"
                        .format(type(input)))

    if not len(input.shape) >= 2:
        raise ValueError("Invalid input shape, we expect BxCx*. Got: {}"
                         .format(input.shape))

    if input.size(0) != target.size(0):
        raise ValueError('Expected input batch_size ({}) to match target batch_size ({}).'
                         .format(input.size(0), target.size(0)))

    n = input.size(0)
    out_size = (n,) + input.size()[2:]
    if target.size()[1:] != input.size()[2:]:
        raise ValueError('Expected target size {}, got {}'.format(
            out_size, target.size()))

    if not input.device == target.device:
        raise ValueError(
            "input and target must be in the same device. Got: {} and {}" .format(
                input.device, target.device))

    # compute softmax over the classes axis
    input_soft: torch.Tensor = F.softmax(input, dim=1) + eps

    # create the labels one hot tensor
    target_one_hot: torch.Tensor = one_hot(
        target, num_classes=input.shape[1],
        device=input.device, dtype=input.dtype)

    # compute the actual focal loss
    weight = torch.pow(-input_soft + 1., gamma)

    focal = -alpha * weight * torch.log(input_soft)
    loss_tmp = torch.sum(target_one_hot * focal, dim=1)

    if reduction == 'none':
        loss = loss_tmp
    elif reduction == 'mean':
        loss = torch.mean(loss_tmp)
    elif reduction == 'sum':
        loss = torch.sum(loss_tmp)
    else:
        raise NotImplementedError("Invalid reduction mode: {}"
                                  .format(reduction))
    return loss
'''


class FocalLoss(nn.Module):
    r"""Criterion that computes Focal loss.

    According to :cite:`lin2018focal`, the Focal loss is computed as follows:

    .. math::

        \text{FL}(p_t) = -\alpha_t (1 - p_t)^{\gamma} \, \text{log}(p_t)

    Where:
       - :math:`p_t` is the model's estimated probability for each class.

    Args:
        alpha (float): Weighting factor :math:`\alpha \in [0, 1]`.
        gamma (float, optional): Focusing parameter :math:`\gamma >= 0`. Default 2.
        reduction (str, optional): Specifies the reduction to apply to the
         output: ‘none’ | ‘mean’ | ‘sum’. ‘none’: no reduction will be applied,
         ‘mean’: the sum of the output will be divided by the number of elements
         in the output, ‘sum’: the output will be summed. Default: ‘none’.
        eps (float, optional): Scalar to enforce numerical stabiliy. Default: 1e-8.

    Shape:
        - Input: :math:`(N, C, *)` where C = number of classes.
        - Target: :math:`(N, *)` where each value is
          :math:`0 ≤ targets[i] ≤ C−1`.

    Example:
        >>> N = 5  # num_classes
        >>> kwargs = {"alpha": 0.5, "gamma": 2.0, "reduction": 'mean'}
        >>> criterion = FocalLoss(**kwargs)
        >>> input = torch.randn(1, N, 3, 5, requires_grad=True)
        >>> target = torch.empty(1, 3, 5, dtype=torch.long).random_(N)
        >>> output = criterion(input, target)
        >>> output.backward()
    """

    def __init__(self, alpha: list = None, gamma: float = 2.0,
                 reduction: str = 'none', ignore_index: int = 9999,
                 eps: float = 1e-8) -> None:
        super(FocalLoss, self).__init__()
        self.alpha: list = alpha
        self.gamma: float = gamma
        self.reduction: str = reduction
        self.ignore_index: int = ignore_index
        self.eps: float = eps

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return focal_loss(input, target, self.alpha, self.gamma, self.reduction,
                          self.ignore_index, self.eps)


''' original version
class FocalLoss(nn.Module):
    r"""Criterion that computes Focal loss.

    According to :cite:`lin2018focal`, the Focal loss is computed as follows:

    .. math::

        \text{FL}(p_t) = -\alpha_t (1 - p_t)^{\gamma} \, \text{log}(p_t)

    Where:
       - :math:`p_t` is the model's estimated probability for each class.

    Args:
        alpha (float): Weighting factor :math:`\alpha \in [0, 1]`.
        gamma (float, optional): Focusing parameter :math:`\gamma >= 0`. Default 2.
        reduction (str, optional): Specifies the reduction to apply to the
         output: ‘none’ | ‘mean’ | ‘sum’. ‘none’: no reduction will be applied,
         ‘mean’: the sum of the output will be divided by the number of elements
         in the output, ‘sum’: the output will be summed. Default: ‘none’.
        eps (float, optional): Scalar to enforce numerical stabiliy. Default: 1e-8.

    Shape:
        - Input: :math:`(N, C, *)` where C = number of classes.
        - Target: :math:`(N, *)` where each value is
          :math:`0 ≤ targets[i] ≤ C−1`.

    Example:
        >>> N = 5  # num_classes
        >>> kwargs = {"alpha": 0.5, "gamma": 2.0, "reduction": 'mean'}
        >>> criterion = FocalLoss(**kwargs)
        >>> input = torch.randn(1, N, 3, 5, requires_grad=True)
        >>> target = torch.empty(1, 3, 5, dtype=torch.long).random_(N)
        >>> output = criterion(input, target)
        >>> output.backward()
    """

    def __init__(self, alpha: float, gamma: float = 2.0,
                 reduction: str = 'none', eps: float = 1e-8) -> None:
        super(FocalLoss, self).__init__()
        self.alpha: float = alpha
        self.gamma: float = gamma
        self.reduction: str = reduction
        self.eps: float = eps

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return focal_loss(input, target, self.alpha, self.gamma, self.reduction, self.eps)
'''


'''def binary_focal_loss_with_logits(
        input: torch.Tensor,
        target: torch.Tensor,
        alpha: float = .25,
        gamma: float = 2.0,
        reduction: str = 'none',
        eps: float = 1e-8) -> torch.Tensor:
    r"""Function that computes Binary Focal loss.

    .. math::

        \text{FL}(p_t) = -\alpha_t (1 - p_t)^{\gamma} \, \text{log}(p_t)

    where:
       - :math:`p_t` is the model's estimated probability for each class.

    Args:
        input (torch.Tensor): input data tensor with shape :math:`(N, 1, *)`.
        target (torch.Tensor): the target tensor with shape :math:`(N, 1, *)`.
        alpha (float): Weighting factor for the rare class :math:`\alpha \in [0, 1]`. Default: 0.25.
        gamma (float): Focusing parameter :math:`\gamma >= 0`. Default: 2.0.
        reduction (str, optional): Specifies the reduction to apply to the. Default: 'none'.
        eps (float): for numerically stability when dividing. Default: 1e-8.

    Returns:
        torch.tensor: the computed loss.

    Examples:
        >>> num_classes = 1
        >>> kwargs = {"alpha": 0.25, "gamma": 2.0, "reduction": 'mean'}
        >>> logits = torch.tensor([[[[6.325]]],[[[5.26]]],[[[87.49]]]])
        >>> labels = torch.tensor([[[1.]],[[1.]],[[0.]]])
        >>> binary_focal_loss_with_logits(logits, labels, **kwargs)
        tensor(4.6052)
    """

    if not isinstance(input, torch.Tensor):
        raise TypeError("Input type is not a torch.Tensor. Got {}"
                        .format(type(input)))

    if not len(input.shape) >= 2:
        raise ValueError("Invalid input shape, we expect BxCx*. Got: {}"
                         .format(input.shape))

    if input.size(0) != target.size(0):
        raise ValueError('Expected input batch_size ({}) to match target batch_size ({}).'
                         .format(input.size(0), target.size(0)))

    probs = torch.sigmoid(input)
    target = target.unsqueeze(dim=1)
    loss_tmp = -alpha * torch.pow((1. - probs), gamma) * target * torch.log(probs + eps) \
               - (1 - alpha) * torch.pow(probs, gamma) * (1. - target) * torch.log(1. - probs + eps)
    loss_tmp = loss_tmp.squeeze(dim=1)

    if reduction == 'none':
        loss = loss_tmp
    elif reduction == 'mean':
        loss = torch.mean(loss_tmp)
    elif reduction == 'sum':
        loss = torch.sum(loss_tmp)
    else:
        raise NotImplementedError("Invalid reduction mode: {}"
                                  .format(reduction))
    return loss



class BinaryFocalLossWithLogits(nn.Module):
    r"""Criterion that computes Focal loss.

    According to :cite:`lin2017focal`, the Focal loss is computed as follows:

    .. math::

        \text{FL}(p_t) = -\alpha_t (1 - p_t)^{\gamma} \, \text{log}(p_t)

    where:
       - :math:`p_t` is the model's estimated probability for each class.

    Args:
        alpha (float): Weighting factor for the rare class :math:`\alpha \in [0, 1]`.
        gamma (float): Focusing parameter :math:`\gamma >= 0`.
        reduction (str, optional): Specifies the reduction to apply to the
         output: ‘none’ | ‘mean’ | ‘sum’. ‘none’: no reduction will be applied,
         ‘mean’: the sum of the output will be divided by the number of elements
         in the output, ‘sum’: the output will be summed. Default: ‘none’.

    Shape:
        - Input: :math:`(N, 1, *)`.
        - Target: :math:`(N, 1, *)`.

    Examples:
        >>> N = 1  # num_classes
        >>> kwargs = {"alpha": 0.25, "gamma": 2.0, "reduction": 'mean'}
        >>> loss = BinaryFocalLossWithLogits(**kwargs)
        >>> input = torch.randn(1, N, 3, 5, requires_grad=True)
        >>> target = torch.empty(1, 3, 5, dtype=torch.long).random_(N)
        >>> output = loss(input, target)
        >>> output.backward()
    """

    def __init__(self, alpha: float, gamma: float = 2.0,
                 reduction: str = 'none') -> None:
        super(BinaryFocalLossWithLogits, self).__init__()
        self.alpha: float = alpha
        self.gamma: float = gamma
        self.reduction: str = reduction
        self.eps: float = 1e-8

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return binary_focal_loss_with_logits(
            input, target, self.alpha, self.gamma, self.reduction, self.eps)'''


############################# LOVASZ-SOFTMAX LOSS ##############################


def lovasz_softmax_3d(probas, labels, classes='present', per_image=False, ignore=None):
    """
    Multi-class Lovasz-Softmax loss
      probas: [B, C, D, H, W] Variable, class probabilities at each prediction (between 0 and 1).
              Interpreted as binary (sigmoid) output with outputs of size [B, D, H, W].
      labels: [B, D, H, W] Tensor, ground truth labels (between 0 and C - 1)
      classes: 'all' for all, 'present' for classes present in labels, or a list of classes to average.
      per_image: compute the loss per image instead of per batch
      ignore: void class labels
    """
    # compute sigmoid activation
    #probas = F.softmax(probas, dim=1)
    probas = probas.log_softmax(dim=1).exp()

    if per_image:
        from LovaszSoftmax.pytorch import mean
        loss = mean(lovasz_losses.lovasz_softmax_flat(*flatten_probas_3d(prob.unsqueeze(0), lab.unsqueeze(0), ignore), classes=classes)
                    for prob, lab in zip(probas, labels))
    else:
        loss = lovasz_losses.lovasz_softmax_flat(
            *flatten_probas_3d(probas, labels, ignore), classes=classes)
    return loss


def flatten_probas_3d(probas, labels, ignore=None):
    """
    Flattens predictions in the batch
    """
    if probas.dim() == 4:
        # assumes output of a sigmoid layer
        B, D, H, W = probas.size()
        probas = probas.view(B, 1, D, H, W)
    B, C, D, H, W = probas.size()
    probas = probas.permute(0, 2, 3, 4, 1).contiguous(
    ).view(-1, C)  # B * D * H * W, C = P, C
    labels = labels.contiguous().view(-1)
    if ignore is None:
        return probas, labels
    valid = (labels != ignore)
    vprobas = probas[valid.nonzero().squeeze()]
    vlabels = labels[valid]
    return vprobas, vlabels


class LovaszSoftmax3d(nn.Module):
    """
    Criterion that computes Lovasz-Softmax loss on 3-dimensional samples.
    """

    def __init__(self, classes: str = 'present', per_image=False, ignore=None):
        super(LovaszSoftmax3d, self).__init__()
        self.classes = classes
        self.per_image = per_image
        self.ignore = ignore

    def forward(self, probas: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return lovasz_softmax_3d(
            probas, labels, self.classes, self.per_image, self.ignore)


################### FOCAL & LOVASZ-SOFTMAX LOSS COMBINATIONS ###################


class SumFocalLovasz(nn.Module):
    '''
    Criterion that sums Focal and Lovasz-Softmax loss.
    ignore:     ignore label
    alpha:      focal loss weighting factors for each class
    gamma:      focal loss focusing parameter
    w:          wegthing factor between focal loss and lovasz-softmax loss
    reduction:  focal_loss redution
    '''

    def __init__(self, classes='present', per_image=False, ignore=None,
                 alpha=None, gamma=2.0, reduction='none',
                 eps=1e-8, w=0.5):

        super(SumFocalLovasz, self).__init__()
        self.classes = classes
        self.per_image = per_image
        self.ignore = ignore
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.eps = eps
        self.w = w

        self.focal_loss = FocalLoss(reduction=self.reduction,
                                    ignore_index=self.ignore,
                                    alpha=self.alpha,
                                    gamma=self.gamma,
                                    eps=self.eps)
        self.lovasz_softmax = LovaszSoftmax3d(classes=self.classes,
                                              per_image=self.per_image,
                                              ignore=self.ignore)

    def forward(self, probas: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        loss = self.w * self.focal_loss(probas, labels) + \
            (1-self.w) * self.lovasz_softmax(probas, labels)

        return loss


################################## DICE LOSS ##################################

# https://github.com/qubvel/segmentation_models.pytorch/blob/master/segmentation_models_pytorch/losses/dice.py

def soft_dice_score(
    output: torch.Tensor,
    target: torch.Tensor,
    smooth: float = 0.0,
    eps: float = 1e-7,
    dims=None,
) -> torch.Tensor:
    assert output.size() == target.size()
    if dims is not None:
        intersection = torch.sum(output * target, dim=dims)
        cardinality = torch.sum(output + target, dim=dims)
    else:
        intersection = torch.sum(output * target)
        cardinality = torch.sum(output + target)
    dice_score = (2.0 * intersection + smooth) / \
        (cardinality + smooth).clamp_min(eps)
    return dice_score


def to_tensor(x, dtype=None) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        if dtype is not None:
            x = x.type(dtype)
        return x
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
        if dtype is not None:
            x = x.type(dtype)
        return x
    if isinstance(x, (list, tuple)):
        x = np.array(x)
        x = torch.from_numpy(x)
        if dtype is not None:
            x = x.type(dtype)
        return x


class DiceLoss(_Loss):
    def __init__(
        self,
        classes: Optional[List[int]] = None,
        log_loss: bool = False,
        from_logits: bool = True,
        smooth: float = 0.0,
        ignore_index: Optional[int] = None,
        eps: float = 1e-7,
    ):
        """Dice loss for image segmentation task.
        It supports multiclass case

        Args:
            classes:  List of classes that contribute in loss computation. By default, all channels are included.
            log_loss: If True, loss computed as `- log(dice_coeff)`, otherwise `1 - dice_coeff`
            from_logits: If True, assumes input is raw logits
            smooth: Smoothness constant for dice coefficient (a)
            ignore_index: Label that indicates ignored pixels (does not contribute to loss)
            eps: A small epsilon for numerical stability to avoid zero division error
                (denominator will be always greater or equal to eps)

        Shape
             - **y_pred** - torch.Tensor of shape (N, C, D, H, W)
             - **y_true** - torch.Tensor of shape (N, D, H, W) or (N, C, D, H, W)

        Reference
            https://github.com/BloodAxe/pytorch-toolbelt
        """
        super(DiceLoss, self).__init__()
        if classes is not None:
            classes = to_tensor(classes, dtype=torch.long)

        self.classes = classes
        self.from_logits = from_logits
        self.smooth = smooth
        self.eps = eps
        self.log_loss = log_loss
        self.ignore_index = ignore_index

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:

        assert y_true.size(0) == y_pred.size(0)  # sambe batch size

        if self.from_logits:
            # Apply activations to get [0..1] class probabilities
            # Using Log-Exp as this gives more numerically stable result and does not cause vanishing gradient on
            # extreme values 0 and 1
            y_pred = y_pred.log_softmax(dim=1).exp()

        bs = y_true.size(0)  # batch size
        num_classes = y_pred.size(1)
        dims = (0, 2)  # batch dim and spatial dims

        # "flatten" spatial dimensions
        y_true = y_true.view(bs, -1)
        y_pred = y_pred.view(bs, num_classes, -1)

        if self.ignore_index is not None:
            mask = y_true != self.ignore_index
            y_pred = y_pred * mask.unsqueeze(1)

            # N, D*H*W -> N, D*H*W, C
            y_true = F.one_hot((y_true * mask).to(torch.long), num_classes)
            y_true = y_true.permute(0, 2, 1) * \
                mask.unsqueeze(1)  # N, C, D*H*W
        else:
            y_true = F.one_hot(y_true, num_classes)  # N, D*H*W -> N, D*H*W, C
            y_true = y_true.permute(0, 2, 1)  # N, C, W*H*W

        scores = self.compute_score(y_pred, y_true.type_as(y_pred),
                                    smooth=self.smooth,
                                    eps=self.eps,
                                    dims=dims)

        if self.log_loss:
            loss = -torch.log(scores.clamp_min(self.eps))
        else:
            loss = 1.0 - scores

        # Dice loss is undefined for non-empty classes
        # So we zero contribution of channel that does not have true pixels
        # NOTE: A better workaround would be to use loss term `mean(y_pred)`
        # for this case, however it will be a modified jaccard loss

        mask = y_true.sum(dims) > 0
        loss *= mask.to(loss.dtype)

        if self.classes is not None:
            loss = loss[self.classes]

        return self.aggregate_loss(loss)

    def aggregate_loss(self, loss):
        return loss.mean()

    def compute_score(self, output, target, smooth=0.0, eps=1e-7, dims=None) -> torch.Tensor:
        return soft_dice_score(output, target, smooth, eps, dims)
