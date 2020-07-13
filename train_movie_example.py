""" Train attention model """
import os
import time
import torch
from ignite.engine import Events, Engine
from ignite.handlers import ModelCheckpoint
from ignite.metrics import Metric
from tensorboardX import SummaryWriter
import fire
import numpy as np
from srn.data import SRNDataloader
from srn.model import MainNet, SpatialRegularizationNet, FuseNet

start_time = time.time()

class AverageLoss(Metric):
    """
    Calculates average AverageLossloss.
    """
    def __init__(self, output_transform=lambda x: x):
        self.acc_loss = None
        super().__init__(output_transform=output_transform)

    def reset(self):
        self.acc_loss = []

    def update(self, loss):
        """
        Parameters
        ----------
        loss : scalar
            loss value
        """
        self.acc_loss.append(loss)

    def compute(self):
        return np.mean(self.acc_loss)

def load_checkpoint(model, checkpoint_path, use_device="cpu"):
    checkpoint = torch.load(checkpoint_path, map_location=use_device)
    model.load_state_dict(checkpoint)
    print("Checkpoint is loaded from: ", checkpoint_path)

def on_iteration_completed(engine, writer):
    loss = engine.state.output
    print("Epoch[{}/{}] Iter: {} - Loss: {:.10f}".format(engine.state.epoch,
                                                         engine.state.max_epochs,
                                                         engine.state.iteration,
                                                         loss))

    writer.add_scalar('Train loss', loss, engine.state.iteration)

    if engine.state.iteration % 100 == 0:
        print("[Train] current time: ", time.time() - start_time)

def run(train_data_path="data/train.csv",
        test_data_path="data/test.csv",
        label_path="data/labels.txt",
        images_path="data/images",
        lr=1e-3, epoch=20, set_device="cpu",
        batch_size=16, test_batch_size=16,
        eval_interval=100,
        out_path="data/outs", model_name="mainnet",
        mainnet_ckpt=None, srn_checkpoint=None,
        fuse_checkpoint=None):
    # check model name
    if model_name not in ["mainnet", "att", "srn", "fuse"]:
        raise "The model should be one of 'mainnet', 'att', 'srn' or 'fuse'."
    print("Start training model: ", model_name)

    # device
    # use_cuda = cuda and torch.cuda.is_available()
    device = torch.device(set_device)
    print("Used device: ", device)

    # Load training data
    train_data = SRNDataloader(data_path=train_data_path, label_path=label_path,
                               images_path=images_path, mode="train")
    train_loader = torch.utils.data.DataLoader(train_data,
                                               shuffle=True,
                                               num_workers=2,
                                               batch_size=batch_size,
                                               drop_last=True)

    test_data = SRNDataloader(data_path=test_data_path, label_path=label_path,
                              images_path=images_path, mode="test")
    test_loader = torch.utils.data.DataLoader(test_data,
                                              shuffle=False,
                                              num_workers=2,
                                              batch_size=test_batch_size,
                                              drop_last=True)

    # Load model
    mainnet = MainNet(num_classes=train_data.get_num_classes())
    if mainnet_ckpt is not None:
        load_checkpoint(mainnet, mainnet_ckpt, use_device=set_device)
    if model_name == "mainnet":
        model = mainnet
    if model_name == "att":
        srn = SpatialRegularizationNet(mainnet)
        if srn_checkpoint is not None:
            load_checkpoint(srn, srn_checkpoint, use_device=set_device)
        model = srn
    elif model_name == "fuse" or model_name == "srn":
        print("Loading Fusenet...")
        srn = SpatialRegularizationNet(mainnet)
        if srn_checkpoint is not None:
            load_checkpoint(srn, srn_checkpoint, use_device=set_device)
        # define mode of fuse model
        if model_name == "srn":
            fuse_mode = "sr"
        else:
            fuse_mode = "finetune"
        print("Fuse mode: ", fuse_mode)
        # init fuse model
        fuse = FuseNet(mainnet, srn, fuse_mode)
        if fuse_checkpoint is not None:
            load_checkpoint(fuse, fuse_checkpoint)
        model = fuse

    model = model.to(device)

    # Init optimizer
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=lr,
                                momentum=0.9,
                                weight_decay=5e-4)
    # Init loss function
    pos_weight = train_data.get_data_pos_weights().to(device)
    print("[DATA] Pos weight: ", pos_weight)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def train_step(_, batch):
        model.train()
        optimizer.zero_grad()

        imgs, targets = batch

        # send to device
        imgs = imgs.to(device)
        targets = targets.to(device)

        # loss
        if fuse_mode != "finetune":
            y_hat = model(imgs)
            loss = loss_fn(y_hat, targets)
        else:
            y_hat, y_att = model(imgs)
            loss_att = loss_fn(y_att, targets)
            loss_fuse = loss_fn(y_hat, targets)
            print("Loss att: {:.5f}, loss fuse: {:.5f}".format(loss_att, loss_fuse))
            loss = loss_att + loss_fuse
        loss.backward()

        optimizer.step()
        return loss.item()

    def test_step(_, batch):
        model.eval()
        with torch.no_grad():
            imgs, targets = batch

            # send to device
            imgs = imgs.to(device)
            targets = targets.to(device)

            # loss
            if fuse_mode != "finetune":
                y_hat = model(imgs)
                loss = loss_fn(y_hat, targets)
            else:
                y_hat, y_att = model(imgs)
                loss_att = loss_fn(y_att, targets)
                loss_fuse = loss_fn(y_hat, targets)
                loss = loss_att + loss_fuse

        return {'test_loss': loss.item()}

    def test_run(engine, writer):
        if engine.state.iteration % eval_interval == 0:
            print("Evaluating...")
            tester.run(test_loader, max_epochs=1)
            metrics = tester.state.metrics

            test_loss = metrics['test_loss']
            print("Test loss: {:.5f}".format(test_loss))
            writer.add_scalar("Test loss", test_loss, engine.state.iteration)

    trainer = Engine(train_step)

    tester = Engine(test_step)
    m = AverageLoss(output_transform=lambda x: x['test_loss'])
    m.attach(tester, "test_loss")

    # timestamp output path
    checkpoint_name = "{}_{}".format(model_name, str(time.time()))
    timed_out_path = os.path.join(out_path, checkpoint_name)

    # summary handlers
    writer = SummaryWriter(timed_out_path)

    # checkpoint handler
    checkpoint_handler = ModelCheckpoint(timed_out_path, 'checkpoint',
                                         n_saved=30, require_empty=False, create_dir=True)

    # add event handler to trainer
    trainer.add_event_handler(event_name=Events.ITERATION_COMPLETED,
                              handler=on_iteration_completed,
                              writer=writer)
    trainer.add_event_handler(event_name=Events.EPOCH_COMPLETED,
                              handler=checkpoint_handler,
                              to_save={'net': model})
    trainer.add_event_handler(event_name=Events.ITERATION_COMPLETED,
                              handler=test_run,
                              writer=writer)

    # start training
    trainer.run(train_loader, max_epochs=epoch)
    writer.close()

if __name__ == "__main__":
    fire.Fire()
