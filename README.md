# Cut-Paste Consistency Learning

This is the official code repository for the WACV 2023
paper "[Cut-Paste Consistency Learning for Semi-Supervised Lesion Segmentation](https://arxiv.org/abs/2210.00191)".

## Installation

```
python -m virtualenv -p 3.6 env
source env/bin/activate

pip install -r requirements.txt
python setup.py install
```

## Downloads

### Datasets

- IDRiD - [Source](https://idrid.grand-challenge.org/)
- CT-ICH - [Source](https://physionet.org/content/ct-ich/1.3.1/) | [CV folds](https://entuedu-my.sharepoint.com/:f:/g/personal/boonpeng001_e_ntu_edu_sg/EhLCdguGK4tElwI-hyQxU2wB5KcAlQyRGMyejz7RYCUqbg?e=IHYnPT)

### Pretrained weights (PyTorch)
Example model checkpoints for the lesion segmentation tasks in IDRiD are provided:

| Model                                                                                                                                         | Method                               | AUC-PR (%) |
|-----------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------|------------|
| [IDRiD-MA](https://entuedu-my.sharepoint.com/:f:/g/personal/boonpeng001_e_ntu_edu_sg/Eri3JNgyQrBAqYMhaogecq8Bkwo4iD5bAc8UfJ0mtOKQZw?e=hfYsee) | Cut-Paste Consistency + Mean Teacher | 51.33      |
| [IDRiD-HE](https://entuedu-my.sharepoint.com/:f:/g/personal/boonpeng001_e_ntu_edu_sg/EsH4KXAB0q1FvOHgdFfT8sIBVxnWpOre99nk6X5H7H0uPw?e=aTq0HN) | Cut-Paste Consistency + Mean Teacher | 66.86      |
| [IDRiD-EX](https://entuedu-my.sharepoint.com/:f:/g/personal/boonpeng001_e_ntu_edu_sg/EhzV9OecJkxEvMCPR-HRWhwBPvMfyehLxBS8htwvqwwVaA?e=BhdZEk) | Cut-Paste Consistency + Mean Teacher | 88.70      |
| [IDRiD-SE](https://entuedu-my.sharepoint.com/:f:/g/personal/boonpeng001_e_ntu_edu_sg/EhWGohbpS9FPm5hPkoMeGrABjYCgkaf2dkcFNmH7kZidFg?e=Jr7I3d) | Cut-Paste Consistency + Mean Teacher | 79.53      |

## Training

List of supported datasets and learning methods:

| data_module                    | model           | Description               |
|--------------------------------|-----------------|---------------------------|
| `idrid`, `ich`                 | `unet`          | Supervised baseline       |
| `idrid-base-cp`, `ich-base-cp` | `unet-pseudo`   | Cut-paste baseline        |
| `idrid-st`, `ich-st`           | `unet-pseudo`   | Self-training             |
| `idrid-st-cp`, `ich-st-cp`     | `unet-pseudo`   | Self-training + cut-paste |
| `idrid-semi`, `ich-semi`       | `unet-mt`       | Mean Teacher              |
| `idrid-semi`, `ich-semi`       | `unet-classmix` | ClassMix consistency      |
| `idrid-semi`, `ich-semi`       | `unet-cutmix`   | CutMix consistency        |
| `idrid-cp`, `ich-cp`           | `unet-cp`       | Cut-paste consistency     |

Type `python main.py <data_module> <model> --help` in the console for more details.

Example of cut-paste consistency learning on IDRiD-MA:

```
python main.py \
    idrid-cp \
    unet-cp \
    --unlabeled_weight 0.01 \
    --mean_teacher \
    --base_ema 0.996 \
    --seed 42 \
    --num_workers 5 \
    --batch_size 5 \
    --synth_split 0.4 \
    --num_synth 300 \
    --mask_blur gaussian \
    --background_blur gaussian	\
    --img_match \
    --val_split 0.1 \
    --data_dir data/IDRiD \
    --gpus [0] \
    --max_epochs 500 \
    --check_val_every_n_epoch 1 \
    --early_stopping_patience -1 \
    --log_every_n_steps 10 \
    --learning_rate 6e-4 \
    --warmup_epochs 10 \
    --optimizer adamw \
    --weight_decay 1e-5 \
    --lr_scheduler cosine \
    --num_layers 5 \
    --features_start 64 \
    --preprocess resize \
    --size 512 \
    --inference_mode resize \
    --inference_size 512 \
    --checkpoint_monitor "val/aupr" \
    --do_train \
    --num_sanity_val_steps 0 \
    --pos_weight 6.84 \
    --default_root_dir "model/idrid-MA" \
    --task_id MA

```

Example of cut-paste consistency learning on CT-ICH:

```
python main.py \
    ich-cp \
    unet-cp \
    --unlabeled_weight 0.1 \
    --mean_teacher \
    --base_ema 0.996 \
    --seed 42 \
    --num_workers 5 \
    --batch_size 8 \
    --labeled_split 0.7 \
    --synth_split 0.4 \
    --img_match \
    --mask_blur gaussian \
    --background_blur none \
    --data_dir "data/CT-ICH/data/fold-1" \
    --default_root_dir "model/ich" \
    --gpus [0] \
    --max_epochs 50 \
    --check_val_every_n_epoch -1 \
    --early_stopping_patience -1 \
    --log_every_n_steps 10 \
    --learning_rate 3e-5 \
    --warmup_epochs 10 \
    --optimizer adamw \
    --weight_decay 1e-5 \
    --lr_scheduler cosine \
    --num_layers 5 \
    --features_start 64 \
    --input_channels 1 \
    --preprocess resize \
    --size 512 \
    --inference_mode resize \
    --inference_size 512 \
    --do_train \
    --do_test \
    --disable_aupr \
    --num_sanity_val_steps 0 \
    --pos_weight 7.08 
```

## Evaluation

Example of evaluating a trained model on IDRiD-MA:

```
python main.py \
    idrid \
    unet \
    --num_workers 1 \
    --data_dir "data/IDRiD" \
    --num_layers 5 \
    --features_start 64 \
    --inference_mode resize \
    --inference_size 512 \
    --do_test \
    --aupr_in_cpu \
    --batch_size 1 \
    --gpus 1 \
    --default_root_dir "model/test/IDRiD-MA" \
    --task_id MA \
    --resume_from_checkpoint "model/IDRiD-MA/checkpoint.ckpt"
```

## Citation
```
@misc{yap2022cutpaste,
      title={Cut-Paste Consistency Learning for Semi-Supervised Lesion Segmentation}, 
      author={Boon Peng Yap and Beng Koon Ng},
      year={2022},
      eprint={2210.00191},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
