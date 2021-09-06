# Process

The scripts to get the results in the report are:

- `l2_dist_expe.py` for the dist to true weights as a function of $\kappa$,
- `perfos_by_kappa.sh` for the performances for different $\kappa$'s, which uses sequentially:
	 - `make_weights_for_cifar.py`
	 - `learn_weighted_cifar.py`
	 - `learn_cifar.py`

Results can be derived with the command:
```
for i in {0..7}
do
	bash perfos_by_kappa.sh ${i}
done
```

Analysis of the results:
```
results_summary.py
plot_dist_over_class.py
```
Imports:
```
gen_biased_datasets.py
custom_resnet.py
solve_weights.py
utils.py
```
