# Move to script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR" || exit 1


type='float'
dist_fn='l2'
data='data/sift/sift_base.fbin'
query='data/sift/sift_query.fbin'
index_prefix='data/sift/insert_delete/index'
result='data/sift/insert_delete/res'

points_to_skip=0
beginning_index_size=600000
pts_per_checkpoint=2000
max_points_to_insert=800000
points_to_delete_from_beginning=200000
deletes_after=0

thr=64
norm=1

iteration=0
interval=5000
start_idx=0
end_idx=0

index=${index_prefix}-insert_delete-iter
gt_file=data/sift/gt100_learn-insert_delete-iter


while [ $iteration -le 0 ]; do

start_idx=$((points_to_skip + iteration * interval))
end_idx=$((start_idx + max_points_to_insert))

index=${index_prefix}-from-${start_idx}-to-${end_idx}-delete-${points_to_delete_from_beginning}
gt_file_before=${gt_file}
gt_file=data/sift/insert_delete/gt100_learn-from-${start_idx}-to-${end_idx}-delete-${points_to_delete_from_beginning}

echo "\n-----iteration: $iteration-----"
echo "------index build------";
../apps/test_insert_deletes_consolidate  --data_type ${type} \
                                        --index_path_prefix ${index_prefix} \
                                        --dist_fn ${dist_fn} \
                                        --data_path ${data} \
                                        --points_to_skip ${start_idx} \
                                        --beginning_index_size ${beginning_index_size} \
                                        --points_per_checkpoint ${pts_per_checkpoint} \
                                        --checkpoints_per_snapshot 0 \
                                        --points_to_delete_from_beginning ${points_to_delete_from_beginning} \
                                        -R 32 -L 100 --alpha 1.2 -T ${thr} \
                                        --max_points_to_insert ${max_points_to_insert} \
                                        --start_deletes_after ${deletes_after} \
                                        --do_concurrent true --start_point_norm 1;


echo ${index}


# echo "\n------compute gt------"
# ./apps/utils/compute_groundtruth --data_type ${type} --dist_fn ${dist_fn} \
#                             --base_file ${index}.data  --query_file ${query}  \
#                             --K 100 --gt_file ${gt_file} \
#                             #--tags_file  ${index_prefix}.tags

# echo "\n------index search------"
# ./apps/search_memory_index  --data_type ${type} --dist_fn ${dist_fn} \
#                             --index_path_prefix ${index} --result_path ${result} \
#                             --query_file ${query}  --gt_file ${gt_file}  \
#                             -K 10 -L 20 40 60 80 100  \
#                             --dynamic false \
#                             #--tags 1  -T ${thr}






    iteration=$((iteration + 1))  
done



