python scripts/train_sf3net.py --use_wandb --save_model --gpus 5 \
                --model_name SF3Net \
                --epochs 100 --batch_size 8 --optimizer AdamW --lr 0.0001 --weight_decay 0 \
                --transform_config ./configs/sf3net_base.yaml --roi_method rand_2 \
                --dataset_root '../../data/' --data RWSteg/Spatial_30000_to_PillowJPG/QT95_1_Subsample_420/ --stego_method UERD_0.4 \
                --wandb_project 2025_SCI_RealWorldSteganalysis

python scripts/train_sf3net.py --use_wandb --save_model --gpus 5 \
                --model_name SF3Net \
                --epochs 100 --batch_size 8 --optimizer AdamW --lr 0.0001 --weight_decay 0 \
                --transform_config ./configs/sf3net_base.yaml --roi_method rand_2 \
                --dataset_root '../../data/' --data RWSteg/Spatial_30000_to_PillowJPG/QT95_2_Subsample_420/ --stego_method UERD_0.4 \
                --wandb_project 2025_SCI_RealWorldSteganalysis

python scripts/train_sf3net.py --use_wandb --save_model --gpus 5 \
                --model_name SF3Net \
                --epochs 100 --batch_size 8 --optimizer AdamW --lr 0.0001 --weight_decay 0 \
                --transform_config ./configs/sf3net_base.yaml --roi_method rand_2 \
                --dataset_root '../../data/' --data RWSteg/Spatial_30000_to_PillowJPG/QT95_4_Subsample_420/ --stego_method UERD_0.4 \
                --wandb_project 2025_SCI_RealWorldSteganalysis