CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch \
--nproc_per_node=8 --master_port 11190 main_byol.py --dataset Kin400RepreLMDB --split 1 \
--n_classes 101 --batch_size 128 --sample_duration 16 \
--model_name r21d_byol --model_depth 1 --ft_begin_index 0 \
--lmdb_path "/dockerdata/yujiazhang/dataset/frame_kinetics_400_mmlab_1f_320_lmdb/lmdb_kin400.lmdb" \
--annotation_path "data_process/kin400_mmlab_labels" \
--result_path "results_kin400_r21d_bsor_rotaug_bs128_lr3e2_wd5e4_bvsvtvp0.1v1v1v1v1v1_mlp_proj_epoch300" \
--n_epochs 300 --learning_rate 0.09 --weight_decay 5e-4 \
--sample_size 112 --n_workers 6 --task loss_com --optimizer sgd --loss_weight 0.1 1 1 1 1

CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch \
--nproc_per_node=2 --master_port 11190 main_ft_mp.py --dataset UcfFineTune --split 1 \
--n_classes 101 --n_finetune_classes 101 --batch_size 60 --sample_duration 16 \
--model_name r21d_byol --model_depth 1 \
--frame_dir "/dockerdata/yujiazhang/dataset/UCF_101_1f_256" \
--annotation_path "data_process/UCF101_labels" \
--result_path "results_kin400_r21d_bsor_rotaug_bs128_lr3e2_wd5e4_bvsvtvp0.1v1v1v1v1v1_mlp_proj_epoch300" \
--pretrained_path "results_kin400_r21d_bsor_rotaug_bs128_lr3e2_wd5e4_bvsvtvp0.1v1v1v1v1v1_mlp_proj_epoch300/Kin400RepreLMDB/loss_com/save_300.pth" \
--n_epochs 100 --learning_rate 0.02 --weight_decay 5e-4 \
--sample_size 112 --n_workers 6 --task "ft_all" --optimizer sgd --transform_mode "img" \
--pb_rate 4

CUDA_VISIBLE_DEVICES=0 python test.py --dataset UcfFineTune --split 1 \
--n_classes 101 --n_finetune_classes 101 \
--batch_size 1 --sample_duration 16 \
--model_name r21d_byol --model_depth 1 --ft_begin_index 5 \
--frame_dir "/dockerdata/yujiazhang/dataset/UCF_101_1f_256" \
--annotation_path "data_process/UCF101_labels" \
--result_path "results_kin400_r21d_bsor_rotaug_bs128_lr3e2_wd5e4_bvsvtvp0.1v1v1v1v1v1_mlp_proj_epoch300" \
--sample_size 112 --n_workers 6 --task "test" --pb_rate 4 --transform_mode "img_test" --t_ft_task "ft_all"
