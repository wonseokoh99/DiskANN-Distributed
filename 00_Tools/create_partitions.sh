

python create_partitions.py \
    --data_file ../01_Index_files/sift/index.in-memory-index.data \
    --partition_file ../01_Index_files/sift/index.in-memory-index.partition8 \
    --num_partitions 8 \
    --dtype float \
    --size_t_bytes 4 \
    --dimension 128