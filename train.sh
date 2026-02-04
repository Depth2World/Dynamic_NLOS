pip install imageio[ffmpeg] pip install imageio[pyav] -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install ipdb scikit-learn tensorboard timm -i https://pypi.tuna.tsinghua.edu.cn/simple



torchrun --nproc_per_node=4 --master_port 29504 train.py \
--model_name 'F3_VIT_COM2_S2PDP_150_15F_rsd6_D1_woclamp' \
--train_bacth_size 1 \
--test_bacth_size 1 \
--num_epoch 50 \
--model_dir '/data/yueli/nlos_video_output/allF24_NDP_v3_com2/' \
--train_filter_path '/data/yueli/dataset/NLOS_bike_video_32frame_3dhelix_complicated2/150train_15F_pathf.txt' \
--train_total_path '/data/yueli/dataset/NLOS_bike_video_32frame_3dhelix_complicated2/150train_15F_path.txt' \
--test_filter_path '/data/yueli/dataset/NLOS_bike_video_32frame_3dhelix_complicated2/test_24F_pathf.txt' \
--test_total_path '/data/yueli/dataset/NLOS_bike_video_32frame_3dhelix_complicated2/test_24F_path.txt' \
--num_save 300 \
--mask
