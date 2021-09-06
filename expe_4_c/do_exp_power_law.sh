alpha="0.75"
i=$1
db_name=$2
kappa=$3 # 10 1 "0.1"
# alpha=$2
obs_per_bin="obs_per_bin/alpha_${alpha}.txt"
python dist_over_classes.py --no_plot --alpha ${alpha} \
    --outfile obs_per_bin/alpha_${alpha}.txt

# for db_name in cifar10 cifar100
# do
    # kappa = 0.5 gives 390K ones in omega (50K x 8 = 400K params), 9.8K zeros
    # for kappa in "0.001" "0.01" "0.1" "0.2"
    # do
        basename=${db_name}_kappa_${kappa}_alpha_${alpha}_ite$i
        out_sample=sample_inds/${basename}.txt
        out_weights=weights/${basename}.txt
        out_concat_weights=weights/${basename}_concat.txt

        python get_image_samples.py --hsv_colors \
          --db_name ${db_name} --kappa ${kappa} \
          --out_sample ${out_sample} --obs_per_bin ${obs_per_bin}

        python make_weights_from_samples.py --hsv_colors \
          --db_name ${db_name} --in_sample ${out_sample} \
          --out_weights ${out_weights} \
          --out_concat_weights ${out_concat_weights} --long_tail_reweight \
					--n_iter 4000


        python learn_weighted_cifar.py --db_name ${db_name} \
                --model_save_name ${basename} --tb_logs_name ${basename} \
                --weights_to_load ${out_weights}

        # python learn_weighted_cifar.py --db_name ${db_name} \
        #         --model_save_name ${basename}_concat \
        #         --tb_logs_name ${basename}_concat \
        #         --weights_to_load ${out_concat_weights}
#     done
# done
