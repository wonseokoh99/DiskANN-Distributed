type='float'
dist_fn='l2'
data='${pwd}/../02_Datasets/sift/sift_base.fbin'
query='../02_Datasets/sift/sift_query.fbin'
index_prefix='../03_Simulation_data/01_index_files/sift/index'
result='../02_Datasets/sift/checkpoint/res'

points_to_skip=200000
beginning_index_size=600000
pts_per_checkpoint=0
max_points_to_insert=600000

thr=64
norm=1

iteration=0
interval=5000
start_idx=0
end_idx=0

while [ $iteration -le 0 ]; do

start_idx=$((points_to_skip + iteration * interval))
end_idx=$((start_idx + max_points_to_insert))

index=${index_prefix}-from-${start_idx}-to-${end_idx}
gt_file=../03_Simulation_data/02_Ground_truth/sift/checkpoint/gt100_learn-from-${start_idx}-to-${end_idx}

echo "\n-----iteration: $iteration-----"
echo "------index build------"
./apps/build_memory_index_offset  --data_type ${type} \
                                        --index_path_prefix ${index_prefix} \
                                        --dist_fn ${dist_fn} \
                                        --data_path ${data} \
                                        --points_to_skip ${start_idx} \
                                        --beginning_index_size ${beginning_index_size} \
                                        --points_per_checkpoint ${pts_per_checkpoint} \
                                        --checkpoints_per_snapshot 0 \
                                        -R 32 -L 100 --alpha 1.2 -T ${thr} \
                                        --max_points_to_insert ${max_points_to_insert} \
                                        --start_point_norm ${norm}
echo ${index}


echo "\n------compute gt------"
./apps/utils/compute_groundtruth --data_type ${type} --dist_fn ${dist_fn} \
                            --base_file ${index}.data  --query_file ${query}  \
                            --K 100 --gt_file ${gt_file} \
                            #--tags_file  ${index_prefix}.tags

echo "\n------index search------"
./apps/search_memory_index  --data_type ${type} --dist_fn ${dist_fn} \
                            --index_path_prefix ${index} --result_path ${result} \
                            --query_file ${query}  --gt_file ${gt_file}  \
                            -K 10 -L 20 40 60 80 100  \
                            --dynamic false \
                            #--tags 1  -T ${thr}






    iteration=$((iteration + 1))  
done



