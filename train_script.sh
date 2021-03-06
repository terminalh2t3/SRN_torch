    python train.py\
        --train_data_path data/custom\
        --images_path /home/lab/custom_data/tag_suggestion/classes10k/images\
        --max_num_testing 10000\
        --dataset_name "5k"\
        --lr 1e-5\
        --epoch 4\
        --set_device "cuda:1"\
        --batch_size 16\
        --test_batch_size 16\
        --eval_interval 500\
        --out_path "data/custom/5k/finetune_v2"\
        --model_name "mainnet"\
        --mainnet_ckpt "data/custom/5k/finetune_v2/mainnet_1581701176.0297775/checkpoint_net_46146.pth"

    python train.py\
        --train_data_path data/custom\
        --images_path /home/lab/custom_data/tag_suggestion/classes10k/images\
        --max_num_testing 10000\
        --dataset_name "5k"\
        --lr 1e-5\
        --epoch 4\
        --set_device "cuda:1"\
        --batch_size 32\
        --test_batch_size 16\
        --eval_interval 1000\
        --out_path "data/custom/5k/finetune_v2"\
        --model_name "att"\
        --mainnet_ckpt "data/custom/5k/finetune_v2/mainnet_1581701176.0297775/checkpoint_net_46146.pth"
