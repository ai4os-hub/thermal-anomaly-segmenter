"""
Classes / Functions for SegFormer model training
"""
import pytorch_lightning as pl
import segmentation_models_pytorch as smp
import torch
from fsspec.core import url_to_fs
from pytorch_lightning.callbacks import ModelCheckpoint
from torch import nn
from transformers import SegformerForSemanticSegmentation


class HfModelCheckpoint(ModelCheckpoint):
    """
    Checkpoint callback that saves the model and the model weights
    to the same directory.

    Note: This is required because the SegformerForSemanticSegmentation model
     from the transformers library does not save correctly using the
     ModelCheckpoint callback provided by PyTorch Lightning.
    """

    def _save_checkpoint(self, trainer: "pl.Trainer", filepath: str) -> None:
        """
        Save checkpoint

        :param trainer: instance of PyTorch Lightning trainer
        :param filepath: path to checkpoint file
        """
        # Save checkpoint using PyTorch Lightning
        super()._save_checkpoint(trainer, filepath)

        # Save model weights to the same directory using HuggingFace
        # transformers library checkpointing
        hf_save_dir = filepath + ".dir"
        if trainer.is_global_zero:
            trainer.lightning_module.model.save_pretrained(hf_save_dir)

    def _remove_checkpoint(self, trainer: "pl.Trainer", filepath: str) -> None:
        """
        Make sure that the checkpoints are also removed correctly

        :param trainer: instance of PyTorch Lightning trainer
        :param filepath: path to checkpoint file
        """
        # Remove checkpoint using PyTorch Lightning
        super()._remove_checkpoint(trainer, filepath)

        # Remove any existing huggingface checkpoints too
        hf_save_dir = filepath + ".dir"
        if trainer.is_global_zero:
            fs, _ = url_to_fs(hf_save_dir)
            if fs.exists(hf_save_dir):
                fs.rm(hf_save_dir, recursive=True)


class SegFormer(pl.LightningModule):
    """
    Defines the SegFormer binary semantic segmentation model
    """

    def __init__(
            self, id2label, mean, std,
            lr=9e-05,
            segformer_pretrained_model="nvidia/"
                                       "segformer-b2-finetuned-ade-512-512",
            load_pretrained=True,
            lr_scheduler="poly",
            loss_fn="tversky"):
        """
        Initialize SegFormer model

        :param id2label: dictionary mapping class ids to class names
        :param mean: mean to use for normalization
        :param std: std to use for normalization
        :param lr: initial learning rate
        :param segformer_pretrained_model: name of pretrained weights to use
        :param load_pretrained: boolean whether to load pretrained weights
        :param lr_scheduler: lr_scheduler to use (plateau or poly)
        :param loss_fn: loss function to use (jaccard, dice, tversky, bce).
         Tversky loss can be used by setting loss_fn to "tversky" or
         "tversky_<alpha>_<beta>" where alpha and beta are loss function
         parameters, f.e. "tversky_0.2_0.8" uses alpha=0.2 and beta=0.8.
        """
        super().__init__()

        # Initialize some variables
        self.mean = mean
        self.std = std

        self.lr_scheduler_type = lr_scheduler

        self.save_hyperparameters()

        self.lr = lr

        self.id2label = id2label
        self.background_label = 0

        self.num_classes = len(id2label.keys())
        self.label2id = {v: k for k, v in self.id2label.items()}

        # If load_pretrained is True, load pretrained weights
        if load_pretrained:
            self.model = SegformerForSemanticSegmentation.from_pretrained(
                segformer_pretrained_model,
                return_dict=True,
                num_labels=self.num_classes,
                id2label=self.id2label,
                label2id=self.label2id,
                ignore_mismatched_sizes=True,
                semantic_loss_ignore_index=0,
            )
        else:
            self.model = None

        # Initialize loss function
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

        # Initialize lists for storing step outputs (for metrics calculation)
        self.training_step_output = []
        self.validation_step_output = []
        self.test_step_output = []

    def set_pretrained_segformer(self, segformer_pretrained_model):
        """
        Set pretrained SegFormer model.

        Note: This is required because the SegformerForSemanticSegmentation
          model from the transformers library does not save correctly using
          the default PyTorch Lightning checkpointing. This function can be
          used to replace the model after initialization.

        :param segformer_pretrained_model: path to pretrained model
        """

        # Load pretrained weights
        self.model = SegformerForSemanticSegmentation.from_pretrained(
            segformer_pretrained_model,
            return_dict=True,
            num_labels=self.num_classes,
            id2label=self.id2label,
            label2id=self.label2id,
            ignore_mismatched_sizes=True,
            semantic_loss_ignore_index=0,
        )

    def forward(self, images, masks=None):
        """
        Forward pass

        :param images: batch of images ([batch_size, channels, height, width])
        :param masks: batch of masks (optional)
        :return: prediction outputs
        """

        if masks is not None:
            masks = masks.squeeze(1)
            # Convert masks to long
            masks = masks.long()

        outputs = self.model(pixel_values=images, labels=masks)

        return (outputs)

    def training_step(self, batch, batch_nb):
        """
        Training step for SegFormer model

        :param batch: batch of data
        :param batch_nb: index of batch
        :return: loss
        """

        images, masks = batch[0], batch[1]
        outputs = self(images, masks)

        # Model outputs logits are upsampled to the original image size, the
        # loss calculated as part of the original implementation is not used
        _, logits = outputs.loss, outputs.logits

        # Upsample logits to original image size
        upsampled_logits = nn.functional.interpolate(
            logits,
            size=masks.shape[-2:],
            mode="bilinear",
            align_corners=False
        )

        loss = self.loss_fn(upsampled_logits, masks)

        # Generate binary mask from logits
        pred_mask = (upsampled_logits.sigmoid().squeeze(1) > 0.5).float()

        # Calculate TPs, FPs, FNs and TNs
        tp, fp, fn, tn = smp.metrics.get_stats(
            pred_mask.long().unsqueeze(1), masks.long(), mode="binary"
        )

        # Save step outputs for later metrics calculation
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
        Validation step for SegFormer model

        :param batch: batch of data
        :param batch_nb: index of batch
        :return: loss
        """

        # Unpack batch
        images, masks = batch[0], batch[1]
        outputs = self(images)
        logits = outputs.logits

        # Upsample logits to original image size
        upsampled_logits = nn.functional.interpolate(
            logits,
            size=masks.shape[-2:],
            mode="bilinear",
            align_corners=False
        )

        loss = self.loss_fn(upsampled_logits, masks)

        pred_mask = (upsampled_logits.sigmoid().squeeze(1) > 0.5).float()

        tp, fp, fn, tn = smp.metrics.get_stats(
            pred_mask.long().unsqueeze(1), masks.long(), mode="binary"
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
        Test step for SegFormer model

        :param batch: batch of data
        :param batch_nb: index of batch
        :return: loss
        """

        images, masks = batch[0], batch[1]
        outputs = self(images)
        logits = outputs.logits

        upsampled_logits = nn.functional.interpolate(
            logits,
            size=masks.shape[-2:],
            mode="bilinear",
            align_corners=False
        )

        loss = self.loss_fn(upsampled_logits, masks)

        pred_mask = (upsampled_logits.sigmoid().squeeze(1) > 0.5).float()

        tp, fp, fn, tn = smp.metrics.get_stats(
            pred_mask.long().unsqueeze(1), masks.long(), mode="binary"
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
        Append step outputs to outputs dictionary

        :param outputs: step outputs dictionary
        :param step_outputs: step outputs
        """

        for key in step_outputs:
            if key not in outputs:
                outputs[key] = []
            outputs[key].append(step_outputs[key])

    def configure_optimizers(self):
        """
        Configure optimizer and learning rate scheduler

        :return:
        """

        # Initialize AdamW optimizer
        optimizer = torch.optim.AdamW(
            [p for p in self.parameters() if p.requires_grad],
            lr=self.lr, eps=1e-08
        )
        # Use ReduceLROnPlateau learning rate scheduler

        # Initialize scheduler
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


def load_segformer_model(checkpoint_path: str, **kwargs):
    """
    Load SegFormer model from checkpoint

    Note: This is required because the SegformerForSemanticSegmentation model
      from the transformers library does not save correctly using the default
      PyTorch Lightning checkpointing.

    :param checkpoint_path: path to model checkpoint (".ckpt") file
    :param kwargs: additional arguments
    :return: loaded model
    """

    # Check if cuda is available and set map_location accordingly
    map_location = None
    if not torch.cuda.is_available():
        map_location = torch.device('cpu')

    model = SegFormer.load_from_checkpoint(
        checkpoint_path,
        load_pretrained=True,
        map_location=map_location,
        **kwargs
    )
    model_dir_path = checkpoint_path + ".dir"
    model.model = SegformerForSemanticSegmentation.from_pretrained(
        model_dir_path,
        return_dict=True,
        num_labels=model.num_classes,
        id2label=model.id2label,
        label2id=model.label2id,
        ignore_mismatched_sizes=True,
        semantic_loss_ignore_index=0,
    )
    return model
