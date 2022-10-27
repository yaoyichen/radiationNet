python main.py \
--main_folder "FullYear" \
--sub_folder "FullYear_V01" \
--train_file "./dataset/fullyear_trainset_240_10_useful.nc" \
--train_point_number 15943 \
--test_file "./dataset/fullyear_trainset_240_10_useful.nc" \
--test_point_number 15943 \
--prefix "fullyear_V01" \
--dataset_type "FullYear" \
--loss_type "v01" \
--learning_rate  1e-3  \
--batch_size 10 \
--model_name "LSTM" \
--num_workers 0 \
--num_epochs 600 \
--save_mode "True" \
--save_checkpoint_name "model" \
--save_per_samples 10000 \
--load_model "False" 