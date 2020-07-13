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
from srn.evaluation import AverageLoss, F1
from srn.loss import BalancedBCELoss

start_time = time.time()

def load_checkpoint(model, checkpoint_path, use_device="cpu"):
    checkpoint = torch.load(checkpoint_path, map_location=use_device)
    model.load_state_dict(checkpoint)
    print("Checkpoint is loaded from: ", checkpoint_path)

def on_iteration_completed(engine, writer):
    global start_time
    loss = engine.state.output
    batch_time = time.time() - start_time
    start_time = time.time()
    print("Epoch[{}/{}] Iter: {} - Loss: {:.10f} - Time: {}(s)".format(engine.state.epoch,
                                                                       engine.state.max_epochs,
                                                                       engine.state.iteration,
                                                                       loss,
                                                                       batch_time))

    writer.add_scalar('Train/Train_loss', loss, engine.state.iteration)

def run(train_data_path="data/custom",
        images_path="data/custom/images",
        max_num_training=None,
        max_num_testing=None,
        dataset_name="5k",
        lr=1e-3, epoch=20, set_device="cpu",
        batch_size=16, test_batch_size=16,
        eval_interval=100,
        out_path="data/custom/outs", model_name="mainnet",
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
    train_data = CustomDataset5k(data_path=train_data_path,
                                images_path=images_path,
                                max_num_training=max_num_training,
                                dataset_name=dataset_name, mode="train")
    train_loader = torch.utils.data.DataLoader(train_data,
                                               shuffle=True,
                                               num_workers=4,
                                               batch_size=batch_size,
                                               drop_last=True)

    test_data = CustomDataset5k(data_path=train_data_path,
                               images_path=images_path,
                               max_num_training=max_num_testing,
                               dataset_name=dataset_name, mode="test")
    test_loader = torch.utils.data.DataLoader(test_data,
                                              shuffle=True,
                                              num_workers=4,
                                              batch_size=test_batch_size,
                                              drop_last=True)

    # Load model
    fuse_mode = None
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
        fuse_mode = "finetune"
        if model_name == "srn":
            fuse_mode = "sr"
        print("Fuse mode: ", fuse_mode)
        # init fuse model
        fuse = FuseNet(mainnet, srn, fuse_mode)
        if fuse_checkpoint is not None:
            load_checkpoint(fuse, fuse_checkpoint)
        model = fuse

    model = model.to(device)

    # Init optimizer
    # optimizer = torch.optim.SGD(model.parameters(),
    #                             lr=lr,
    #                             momentum=0.9,
    #                             weight_decay=5e-4)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Init loss function
    # pos_weight = train_data.get_data_pos_weights().to(device) / 4.0
    # pos_weight = torch.ones(train_data.get_num_classes(),).to(device)

    # print("[DATA] Pos weight: ", pos_weight)
    # print("[DATA] Pos weight shape: ", pos_weight.shape)

    # Init loss function
    # loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    weights = [1.5 - 40/500, 0.5 + 40/500]
    loss_fn = BalancedBCELoss(weights)

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

            confidence = torch.sigmoid(y_hat)

        return {
            'test_loss': loss.item(),
            'confidence': confidence.cpu().detach().numpy(),
            'target': targets.cpu().detach().numpy()
        }

    def test_run(engine, writer):
        if engine.state.iteration % eval_interval == 0:
            print("Evaluating...")
            tester.run(test_loader, max_epochs=1)
            metrics = tester.state.metrics

            test_loss = metrics['test_loss']
            recall, precision, f1, num_pos = metrics['f1']
            print("""[Evaluation]
                    Test loss: {:.5f}
                    - F1: {:.2f}
                    - Precision: {:.2f}
                    - Recall: {:.2f}
                    - Num pos: {:.2f}""".format(test_loss, f1, precision, recall, num_pos))

            writer.add_scalar("Eval/Test_loss", test_loss, engine.state.iteration)
            writer.add_scalar("Eval/F1", f1, engine.state.iteration)
            writer.add_scalar("Eval/Recall", recall, engine.state.iteration)
            writer.add_scalar("Eval/Precision", precision, engine.state.iteration)
            writer.add_scalar("Eval/Average_num_positive",
                              num_pos, engine.state.iteration)

    trainer = Engine(train_step)

    tester = Engine(test_step)
    m1 = AverageLoss(output_transform=lambda x: x['test_loss'])
    m1.attach(tester, "test_loss")

    m2 = F1(output_transform=lambda x: (x['target'], x['confidence']))
    m2.attach(tester, "f1")

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
    fire.Fire(run)
