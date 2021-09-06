i=$1 # 8 iterations
n_datasets=5
# db_name=$1
# kappa=$2

mkdir weights models logs figures data

for db_name in cifar100 cifar10 
do
    for kappa in "0.001" "0.01" "0.1" "0.2" 
    do
        basename="${db_name}_kappa${kappa}_ite${i}"
        weight_file="weights/${basename}.txt"
        python make_weights_for_cifar.py --name_db ${db_name} \
                --n_datasets ${n_datasets} --kappa ${kappa} \
                --outfile ${weight_file}

        python learn_weighted_cifar.py --db_name ${db_name} \
                --model_save_name ${basename} --tb_logs_name ${basename} \
                --weights_to_load ${weight_file}
    done

    basename="${db_name}_ite${i}"
    python learn_cifar.py --db_name ${db_name} \
       --model_save_name ${basename} --tb_logs_name ${basename}
done
