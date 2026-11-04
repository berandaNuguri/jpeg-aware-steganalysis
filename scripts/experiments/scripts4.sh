python scripts/train.py --use_wandb --save_model --gpus 3 \
                --model_name TempNet_v53 \
                --epochs 100 --batch_size 8 --optimizer AdamW --lr 0.0001 --weight_decay 0 \
                --transform_config ./configs/base.yaml --roi_method rand_2 \
                --dataset_root '../../data/' --data RWSteg/Spatial_30000_to_PillowJPG/QT90_1_Subsample_420/ --stego_method UERD_0.4 \
                --wandb_project 2025_SCI_RealWorldSteganalysis --suffix Base

python scripts/train.py --use_wandb --save_model --gpus 3 \
                --model_name TempNet_v53 \
                --epochs 100 --batch_size 8 --optimizer AdamW --lr 0.0001 --weight_decay 0 \
                --transform_config ./configs/base.yaml --roi_method rand_2 \
                --dataset_root '../../data/' --data RWSteg/Spatial_30000_to_PillowJPG/QT95_1_Subsample_420/ --stego_method UERD_0.4 \
                --wandb_project 2025_SCI_RealWorldSteganalysis --suffix Base