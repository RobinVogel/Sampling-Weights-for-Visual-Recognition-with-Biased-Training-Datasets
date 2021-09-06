### Process

Use `do_exp.sh 1` where 1 is the iteration number, which uses sequentially:
```
get_image_samples.py
make_weights_from_samples.py
learn_weighted_cifar.py
```
Use `do_exp_power_law.sh 1` where 1 is the iteration number, which does
all of the above after it uses `dist_over_classes.py` to generate the distribution
over biased datasets.

Imports:
```
custom_resnet.py
utils.py
solve_weights.py
```

Scripts to find out how to split the dataa:
```
image_analysis.py
```

Make representations:
```
weights_analysis.py
image_analysis.py
plot_image_samples.py
plot_dist_over_bins.py
plot_dist_over_dims.py
make_montages.sh
```

Scripts to plot the results:
```
results_summary.py
```
