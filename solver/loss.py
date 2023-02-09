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

    def forward(self, pred, pred_128, pred_64, pred_32, target, target_128, target_64, target_32):
        pred = self.sigmoid(pred.squeeze())
        pred_128 = self.sigmoid(pred_128.squeeze())
        pred_64 = self.sigmoid(pred_64.squeeze())
        pred_32 = self.sigmoid(pred_32.squeeze())
        
        soft_dice_loss = self.dice_loss(pred, target) * self.w_dice
        bce_loss = self.focal_loss(pred, target) * self.w_focal
        
        soft_dice_loss_128 = self.dice_loss(pred_128, target_128) * self.w_dice
        bce_loss_128  = self.focal_loss(pred_128, target_128) * self.w_focal
        
        soft_dice_loss_64 = self.dice_loss(pred_64, target_64) * self.w_dice
        bce_loss_64 = self.focal_loss(pred_64, target_64) * self.w_focal
        
        soft_dice_loss_32 = self.dice_loss(pred_32, target_32) * self.w_dice
        bce_loss_32 = self.focal_loss(pred_32, target_32) * self.w_focal

        loss = 0.5*(soft_dice_loss+bce_loss) \
                + 0.3*(soft_dice_loss_128+bce_loss_128) \
                + 0.2*(soft_dice_loss_64+bce_loss_64) \
                + 0.1*(soft_dice_loss_32+bce_loss_32)
        # print(loss)

        return loss