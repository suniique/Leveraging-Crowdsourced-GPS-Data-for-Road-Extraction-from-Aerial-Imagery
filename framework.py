import torch
import torch.nn as nn
from torch.autograd import Variable
import os
from tqdm import tqdm
from utils.metrics import IoU
from loss import dice_bce_loss

class Solver:
    def __init__(self, net, optimizer, loss=dice_bce_loss, metrics=IoU):
        self.net = net.cuda()
        self.net = torch.nn.DataParallel(
            self.net, device_ids=list(range(torch.cuda.device_count()))
        )
        self.optimizer = optimizer
        self.loss = loss()
        self.metrics = metrics()
        self.old_lr = optimizer.param_groups[0]["lr"]

    def set_input(self, img_batch, mask_batch=None, img_id=None):
        self.img = img_batch
        self.mask = mask_batch
        self.img_id = img_id

    def forward(self, volatile=False):
        self.img = Variable(self.img.cuda(), volatile=volatile)
        if self.mask is not None:
            self.mask = Variable(self.mask.cuda(), volatile=volatile)

    def optimize(self):
        self.net.train()
        self.forward()
        self.optimizer.zero_grad()
        pred = self.net.forward(self.img)
        loss = self.loss(self.mask, pred)
        loss.backward()
        self.optimizer.step()
        metrics = self.metrics(self.mask, pred)
        return loss.item(), metrics

    def save_weights(self, path):
        torch.save(self.net.state_dict(), path)

    def load_weights(self, path):
        self.net.load_state_dict(torch.load(path))

    def update_lr(self, new_lr, factor=False):
        if factor:
            new_lr = self.old_lr / new_lr
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = new_lr

        print("==> update learning rate: %f -> %f" % (self.old_lr, new_lr))
        self.old_lr = new_lr

    def test_batch(self):
        self.net.eval()
        self.forward(volatile=True)
        pred = self.net.forward(self.img)
        loss = self.loss(self.mask, pred)
        metrics = self.metrics(self.mask, pred)
        pred = pred.cpu().data.numpy().squeeze(1)
        return pred, loss.item(), metrics

    def pred_one_image(self, image):
        self.net.eval()
        image = image.cuda().unsqueeze(0)
        pred = self.net.forward(image)
        return pred.cpu().data.numpy().squeeze(1).squeeze(0)


class Trainer:
    def __init__(self, *args, **kwargs):
        self.solver = Solver(*args, **kwargs)

    def set_train_dl(self, dataloader):
        self.train_dl = dataloader

    def set_validation_dl(self, dataloader):
        self.validation_dl = dataloader

    def set_test_dl(self, dataloader):
        self.test_dl = dataloader

    def set_save_path(self, save_path):
        self.save_path = save_path

    def fit_one_epoch(self, dataloader, eval=False):
        dataloader_iter = iter(dataloader)
        epoch_loss = 0
        epoch_metrics = 0
        iter_num = len(dataloader_iter)
        progress_bar = tqdm(enumerate(dataloader_iter), total=iter_num)
        for i, (img, mask) in progress_bar:
            self.solver.set_input(img, mask)
            if eval:
                _, iter_loss, iter_metrics = self.solver.test_batch()
            else:
                iter_loss, iter_metrics = self.solver.optimize()
            epoch_loss += iter_loss
            epoch_metrics += iter_metrics
            progress_bar.set_description(
                f'iter: {i} loss: {iter_loss:.4f} metrics: {iter_metrics[3]:.4f}'
            )
        epoch_loss /= iter_num
        epoch_metrics /= iter_num
        return epoch_loss, epoch_metrics

    def fit(self, epochs, no_optim_epochs=10):
        val_best_metrics = 0
        val_best_loss = float("+inf")
        no_optim = 0
        for epoch in range(1, epochs + 1):
            print(f"epoch {epoch}/{epochs}")

            print(f"training")
            train_loss, train_metrics = self.fit_one_epoch(self.train_dl, eval=False)

            print(f"validating")
            val_loss, val_metrics = self.fit_one_epoch(self.validation_dl, eval=True)

            print(f"testing")
            test_loss, test_metrics = self.fit_one_epoch(self.test_dl, eval=True)

            print('epoch finished')
            print(f'train_loss: {train_loss:.4f} train_metrics: {train_metrics}')
            print(f'val_loss: {val_loss:.4f} val_metrics: {val_metrics}')
            print(f'test_loss: {test_loss:.4f} val_metrics: {test_metrics}')
            print()

            if val_metrics[3] > val_best_metrics:
                val_best_metrics = val_metrics[3]
                self.solver.save_weights(os.path.join(self.save_path,
                    f"epoch{epoch}_val{val_metrics[3]:.4f}_test{test_metrics[3]:.4f}.pth"))

            if val_loss < val_best_loss:
                no_optim = 0
                val_best_loss = val_loss
            else:
                no_optim += 1

            if no_optim > no_optim_epochs:
                if self.solver.old_lr < 1e-8:
                    print('early stop at {epoch} epoch')
                    break
                else:
                    no_optim = 0
                    self.solver.update_lr(5.0, factor=True)


class Tester:
    def __init__(self, *args, **kwargs):
        self.solver = Solver(*args, **kwargs)

    def set_validation_dl(self, dataloader):
        self.validation_dl = dataloader

    def predict(self):
        pass


