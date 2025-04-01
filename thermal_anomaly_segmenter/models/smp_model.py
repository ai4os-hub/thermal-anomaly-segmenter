"""
Classes / Functions for any SMP (segmentation_models_pytorch) model training
"""
import pytorch_lightning as pl
import segmentation_models_pytorch as smp
import torch
from torch import nn


class SMPModel(pl.LightningModule):
    """
    Defines an SMP model.
    This is a wrapper around segmentation_models_pytorch model and can be used
    to create any smp model that is supported by segmentation_models_pytorch.
    See pytorch_lightning documentation for more information on the supported
    model architectures: https://github.com/qubvel/segmentation_models.pytorch
    """

    def __init__(
            self, mean, std,
            arch="DeepLabV3Plus",
            encoder_name="resnet101",
            lr=9e-05,
            in_channels=3,
            lr_scheduler="poly",
            loss_fn="tversky"
    ):
        """
        Initialize SMP model

        :param mean: mean to use for normalization
        :param std: std to use for normalization
        :param arch: architecture to use
        :param encoder_name: encoder to use
        :param lr: initial learning rate
        :param in_channels: number of input channels
        :param lr_scheduler: learning rate scheduler to use
        :param loss_fn: loss function to use (jaccard, dice, tversky, bce).
         Tversky loss can be used by setting loss_fn to "tversky" or
         "tversky_<alpha>_<beta>" where alpha and beta are loss function
         parameters, f.e. "tversky_0.2_0.8" uses alpha=0.2 and beta=0.8.
        """
        super().__init__()
        self.mean = mean
        self.std = std

        self.lr_scheduler_type = lr_scheduler

        self.save_hyperparameters()

        self.lr = lr

        self.background_label = 0

        # Load smp model
        self.model = smp.create_model(
            arch,
            encoder_name=encoder_name,
            in_channels=in_channels,
            classes=1,
            encoder_weights="imagenet"
        )

        # Select loss function
        if loss_fn == "jaccard":
            self.loss_fn = smp.losses.JaccardLoss(
                mode="binary", from_logits=True
            )
        elif loss_fn == "dice":
            self.loss_fn = smp.losses.DiceLoss(
                mode="binary", from_logits=True
            )
        elif "tversky" in loss_fn:
            if loss_fn == "tversky":
                alpha = 0.2
                beta = 0.8
            else:
                alpha = float(loss_fn.split("_")[1])
                # Convert to float
                alpha = float(alpha)
                beta = float(loss_fn.split("_")[2])
                beta = float(beta)
            self.loss_fn = smp.losses.TverskyLoss(
                mode="binary", from_logits=True, alpha=alpha, beta=beta
            )
        elif loss_fn == "bce":
            # BCE Loss
            self.loss_fn = nn.BCEWithLogitsLoss()
        else:
            raise Exception("Loss function not implemented")

        self.training_step_output = []
        self.validation_step_output = []
        self.test_step_output = []

    def forward(self, images):
        """
        Forward pass

        :param images: input images
        :return: output logits
        """
        outputs = self.model(images)

        return (outputs)

    def training_step(self, batch, batch_nb):
        """
        Training step

        :param batch: data batch
        :param batch_nb: batch number
        :return: loss
        """

        # Unpack the batch
        images, masks = batch[0], batch[1]

        logits_mask = self(images)

        loss = self.loss_fn(logits_mask, masks)

        prob_mask = logits_mask.sigmoid()

        pred_mask = (prob_mask > 0.5).float()

        tp, fp, fn, tn = smp.metrics.get_stats(
            pred_mask.long(), masks.long(), mode="binary"
        )

        step_output = {
            "loss": loss,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
        }
        self.training_step_output.append(step_output)

        return loss

    def on_train_epoch_end(self):
        """
        Calculate metrics at the end of each training epoch
        """
        self.calculate_log_metrics(stage="training",
                                   outputs=self.training_step_output)
        self.training_step_output.clear()
        torch.cuda.empty_cache()

    def validation_step(self, batch, batch_nb):
        """
        Validation step

        :param batch: data batch
        :param batch_nb: batch number
        :return:
        """

        # Unpack the batch
        images, masks = batch[0], batch[1]

        logits_mask = self(images)

        loss = self.loss_fn(logits_mask, masks)

        prob_mask = logits_mask.sigmoid()

        pred_mask = (prob_mask > 0.5).float()
        tp, fp, fn, tn = smp.metrics.get_stats(
            pred_mask.long(), masks.long(), mode="binary"
        )
        self.log(
            "valid_loss", loss, on_step=True, on_epoch=True, prog_bar=False
        )

        step_output = {
            "loss": loss,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
        }
        self.validation_step_output.append(step_output)

        return loss

    def on_validation_epoch_end(self):
        """
        Calculate metrics at the end of each validation epoch
        """
        self.calculate_log_metrics(stage="valid",
                                   outputs=self.validation_step_output)
        self.validation_step_output.clear()
        torch.cuda.empty_cache()

    def test_step(self, batch, batch_nb):
        """
        Test step

        :param batch: batch of data
        :param batch_nb: batch number
        :return: loss
        """

        images, masks = batch[0], batch[1]

        logits_mask = self(images)

        loss = self.loss_fn(logits_mask, masks)

        prob_mask = logits_mask.sigmoid()

        # loss = F.binary_cross_entropy(prob_mask, mask, weight=weights)

        pred_mask = (prob_mask > 0.5).float()
        tp, fp, fn, tn = smp.metrics.get_stats(
            pred_mask.long(), masks.long(), mode="binary"
        )

        step_output = {
            "loss": loss,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
        }
        self.test_step_output.append(step_output)

        return loss

    def on_test_epoch_end(self):
        """
        Calculate metrics at the end of each test epoch
        """
        self.calculate_log_metrics(stage="test",
                                   outputs=self.test_step_output)
        self.test_step_output.clear()
        torch.cuda.empty_cache()

    def calculate_log_metrics(self, stage, outputs):
        """
        Calculate metrics

        :param stage: stage (training, valid, test)
        :param outputs: list of stored outputs for that stage
        :return: logged metrics
        """
        tp = torch.cat([x["tp"] for x in outputs])
        fp = torch.cat([x["fp"] for x in outputs])
        fn = torch.cat([x["fn"] for x in outputs])
        tn = torch.cat([x["tn"] for x in outputs])

        # Calculate per image metrics
        per_image_iou = smp.metrics.iou_score(
            tp, fp, fn, tn, reduction="micro-imagewise"
        )
        per_image_dice = smp.metrics.f1_score(
            tp, fp, fn, tn, reduction="micro-imagewise"
        )
        per_image_recall = smp.metrics.recall(
            tp, fp, fn, tn, reduction="micro-imagewise"
        )
        per_image_precision = smp.metrics.precision(
            tp, fp, fn, tn, reduction="micro-imagewise"
        )
        per_image_f2 = smp.metrics.fbeta_score(
            tp, fp, fn, tn, beta=2, reduction="micro-imagewise"
        )

        # Calculate data metrics
        dataset_iou = smp.metrics.iou_score(
            tp, fp, fn, tn, reduction="micro"
        )
        dataset_dice = smp.metrics.f1_score(
            tp, fp, fn, tn, reduction="micro"
        )
        dataset_recall = smp.metrics.recall(
            tp, fp, fn, tn, reduction="micro"
        )
        dataset_precision = smp.metrics.precision(
            tp, fp, fn, tn, reduction="micro"
        )
        dataset_f2 = smp.metrics.fbeta_score(
            tp, fp, fn, tn, beta=2, reduction="micro"
        )

        # Log metrics (not displayed in progress bar)
        metrics = {
            f"{stage}_dataset_dice": dataset_dice,
            f"{stage}_dataset_f2": dataset_f2,
            f"{stage}_dataset_iou": dataset_iou,
            f"{stage}_dataset_precision": dataset_precision,
            f"{stage}_dataset_recall": dataset_recall,
            f"{stage}_per_image_dice": per_image_dice,
            f"{stage}_per_image_f2": per_image_f2,
            f"{stage}_per_image_iou": per_image_iou,
            f"{stage}_per_image_precision": per_image_precision,
            f"{stage}_per_image_recall": per_image_recall,
        }
        self.log_dict(metrics, prog_bar=False)

    def append_to_step_outputs(self, outputs, step_outputs):
        """
        Append step outputs to outputs

        :param outputs: list of outputs
        :param step_outputs: step outputs
        :return:
        """

        for key in step_outputs:
            if key not in outputs:
                outputs[key] = []
            outputs[key].append(step_outputs[key])

    def configure_optimizers(self):
        """
        Configure optimizers

        :return: optimizer configuartion
        """

        # Initialize optimizer
        optimizer = torch.optim.AdamW(
            [p for p in self.parameters() if p.requires_grad],
            lr=self.lr, eps=1e-08
        )

        # Initialize learning rate scheduler
        if self.lr_scheduler_type == "plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, factor=0.1, patience=4
            )
            lr_config = {
                'scheduler': scheduler,
                'monitor': 'valid_loss',
                "interval": "epoch"
            }
        elif self.lr_scheduler_type == "poly":
            num_epochs = self.trainer.max_epochs
            scheduler = torch.optim.lr_scheduler.PolynomialLR(
                optimizer, total_iters=num_epochs, power=1.0
            )
            lr_config = {
                'scheduler': scheduler,
                "interval": "epoch"
            }
        else:
            raise NotImplementedError(
                f"Unsupported lr scheduler type: {self.lr_scheduler_type}"
            )

        return {
            'optimizer': optimizer,
            'lr_scheduler': lr_config}
