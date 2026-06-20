import torch
import torch.nn as nn
import time

class Trainer:

    def __init__(
        self,
        model,
        optimizer,
        loss_fn,
        out_path,
        device,
        logger,
        cfg
    ):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.out_path = out_path
        self.device = device
        self.logger = logger
        self.cfg = cfg


    def train_epoch(self, train_dl, epoch):

        self.model.train() # enable dropout

        running_loss = 0.0
        running_correct = 0.0

        for x_batch, y_batch in train_dl:
            # put data to device - GPU - if available
            x_batch = x_batch.to(self.device)
            y_batch = y_batch.float().to(self.device)

            # forward pass
            pred = self.model(x_batch)[:, 0]

            # loss calc
            loss = self.loss_fn(pred, y_batch)

            # gradient accum reset
            self.optimizer.zero_grad()

            # gradients calc
            loss.backward()

            # update params
            self.optimizer.step()

            # log metrics
            running_loss += loss.item() * y_batch.size(0)

            is_correct = ( (pred >= 0.5).float() == y_batch ).float()
            running_correct += is_correct.sum().item()
        
        epoch_loss = running_loss / len(train_dl.dataset)
        epoch_acc = running_correct / len(train_dl.dataset)

        return epoch_loss, epoch_acc
    

    def validate(self, valid_dl):
        self.model.eval() # disable dropout

        running_loss = 0.0
        running_correct = 0.0

        check = True

        with torch.no_grad():
            for x_batch, y_batch in valid_dl:

                # put data to device
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.float().to(self.device)

                # forward pass
                pred = self.model(x_batch)[:, 0]

                # loss calc
                loss = self.loss_fn(pred, y_batch)

                # metrics
                running_loss += loss.item() * y_batch.size(0)

                is_correct = ( (pred>=0.5).float() == y_batch ).float()
                running_correct += is_correct.sum().item()
        
        val_loss = running_loss / len(valid_dl.dataset)
        val_acc = running_correct / len(valid_dl.dataset)

        return val_loss, val_acc


    def train(self, train_dl, valid_dl):
        self.logger.info("=" * 80)
        self.logger.info("TRAINING")
        self.logger.info("=" * 80)

        best_val_acc = 0.0

        history = {
            "train_loss": [],
            "val_loss": [],
            "train_acc": [],
            "val_acc": [],
            "train_time": [],
            "valid_time": [],
            "epoch_time": []
        }

        overall_start_time = time.time()

        for epoch in range(self.cfg['training']['epochs']):
            # training
            train_start_time = time.time()
            train_loss, train_acc = self.train_epoch(train_dl=train_dl, epoch=epoch)
            train_time = time.time() - train_start_time

            # validation
            valid_start_time = time.time()
            val_loss, val_acc = self.validate(valid_dl=valid_dl)
            valid_time = time.time() - valid_start_time

            total_epoch_time = train_time + valid_time

            # save metrics
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['train_acc'].append(train_acc)
            history['val_acc'].append(val_acc)

            # save times
            history['train_time'].append(train_time)
            history['valid_time'].append(valid_time)
            history['epoch_time'].append(total_epoch_time)


            # logging
            self.logger.info(
                f"[Epoch {epoch:03d}/{self.cfg['training']['epochs']:03d}]  |  "
                f'train_loss={train_loss:.3f}  |  '
                f'train_acc={train_acc:.3f}  |  '
                f'valid_loss={val_loss:.3f}  |  '
                f'valid_acc={val_acc:.3f}  |  '
                f'train_time={train_time:.2f}s  |  '
                f'valid_time={valid_time:.2f}s  |  '
                f'epoch_time={total_epoch_time:.2f}s'
            )

            # save model after each epoch
            torch.save(self.model, self.out_path + f'/ep{epoch}-' + self.cfg['checkpointing']['model_name'])
            torch.save(self.model.state_dict(), self.out_path + f'/ep{epoch}-' + self.cfg['checkpointing']['model_weights'])

            # save model if better than current best
            if val_acc > best_val_acc:
                best_val_acc = val_acc

                # torch.save(self.model, cfg['checkpointing']['model_output'])
                torch.save(self.model, self.out_path + '/best-' + self.cfg['checkpointing']['model_name'])
                # torch.save(self.model.state_dict(), cfg['checkpointing']['model_dict_output'])
                torch.save(self.model.state_dict(), self.out_path + '/best-' + self.cfg['checkpointing']['model_weights'])

                self.logger.info(
                    f'    New best model saved!!! '
                    f'(val_acc={val_acc:.3f})'
                )
            # self.logger.info('\n')
        
        overall_time = time.time() - overall_start_time

        self.logger.info('\n')
        self.logger.info('=' * 80)
        self.logger.info('TRAINING COMPLETE')
        self.logger.info('=' * 80)
        self.logger.info(
            f'Total Training Time: '
            f'{overall_time:.2f} seconds'
        )
        self.logger.info('\n')

        return history
