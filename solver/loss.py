import paddle.nn as nn
import paddle
import math
import paddle.nn.functional as F
# from paddle.autograd import Variable

class DiceLoss(nn.Layer):
    def __init__(self, smooth=1.):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, preds, targets):
        # outputs = torch.sigmoid(preds.squeeze())

        numerator = 2 * paddle.sum(preds * targets) + self.smooth
        denominator = paddle.sum(preds ** 2) + paddle.sum(targets ** 2) + self.smooth
        soft_dice_loss = 1 - numerator / denominator

        return soft_dice_loss


class WeightedFocalLoss(nn.Layer):
    "Non weighted version of Focal Loss"
    def __init__(self, alpha=.01, gamma=2):
        super(WeightedFocalLoss, self).__init__()
        self.alpha = paddle.to_tensor([alpha, 1-alpha])
        self.gamma = gamma

    def forward(self, preds, targets):
        BCE_loss = F.binary_cross_entropy(paddle.flatten(preds), paddle.flatten(targets).astype("float32"), reduction='none')
        targets = targets.astype(paddle.compat.long_type)
        # self.alpha = self.alpha.to(preds.device)
        at = self.alpha.gather(paddle.flatten(targets))
        pt = paddle.exp(-BCE_loss)
        F_loss = at*(1-pt)**self.gamma * BCE_loss
        F_loss = F_loss.mean()

        if math.isnan(F_loss) or math.isinf(F_loss):
            F_loss = paddle.zeros(1).to(preds.device)

        return F_loss

class Loss(nn.Layer):
    def __init__(self):
        super(Loss, self).__init__()
        self.alpha = 0.4
        self.dice_loss = DiceLoss()
        self.focal_loss = WeightedFocalLoss()
        self.w_dice = 1.
        self.w_focal = 100.
        self.S_dice = []
        self.S_focal = []
        self.sigmoid = paddle.nn.Sigmoid()

    def forward(self, preds, targets):
        preds = self.sigmoid(preds.squeeze())
        dice_loss = self.dice_loss(preds, targets) * self.w_dice
        focal_loss = self.focal_loss(preds, targets) * self.w_focal

        return dice_loss, focal_loss