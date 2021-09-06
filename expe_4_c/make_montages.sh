n_imgs=20
target_folder="figures/balanced/"
for i in {0..7}
do
  echo "Working on montage "$i
  montage -mode concatenate $(ls ${target_folder}sample_$i/* | head -n ${n_imgs}) -geometry +1+1 \
    "${target_folder}montage_sample_$i.png"
done
montage -mode concatenate -geometry +5+5 \
	-label "" ${target_folder}montage_sample_0.png \
        -label "" ${target_folder}montage_sample_1.png \
        -label "" ${target_folder}montage_sample_2.png \
        -label "" ${target_folder}montage_sample_3.png \
        -label "" ${target_folder}montage_sample_4.png \
        -label "" ${target_folder}montage_sample_5.png \
        -label "" ${target_folder}montage_sample_6.png \
        -label "" ${target_folder}montage_sample_7.png \
        ${target_folder}/res.png
