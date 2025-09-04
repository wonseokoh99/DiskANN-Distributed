// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include <iostream>
#include "utils.h"

void block_convert(std::ifstream &reader, std::ofstream &writer, uint8_t *read_buf, uint8_t *write_buf, size_t npts,
                   size_t ndims)
{
    // bvecs format: [dimension][byte_data][dimension][byte_data]...
    // Each record = 4 bytes (dimension) + ndims bytes (data)
    reader.read((char *)read_buf, npts * (ndims + sizeof(uint32_t)));
    
    for (size_t i = 0; i < npts; i++)
    {
        // Skip the dimension header (4 bytes) and copy only the byte data
        memcpy(write_buf + i * ndims, read_buf + i * (ndims + sizeof(uint32_t)) + sizeof(uint32_t), ndims);
    }
    writer.write((char *)write_buf, npts * ndims);
}

int main(int argc, char **argv)
{
    if (argc != 3)
    {
        std::cout << argv[0] << " input_bvecs output_bin" << std::endl;
        exit(-1);
    }
    
    std::ifstream reader(argv[1], std::ios::binary | std::ios::ate);
    size_t fsize = reader.tellg();
    reader.seekg(0, std::ios::beg);

    // Read first dimension value
    uint32_t ndims_u32;
    reader.read((char *)&ndims_u32, sizeof(uint32_t));
    reader.seekg(0, std::ios::beg);
    size_t ndims = (size_t)ndims_u32;
    
    // Calculate number of points: each point = 4 bytes (dimension) + ndims bytes (data)
    size_t npts = fsize / (ndims + sizeof(uint32_t));
    std::cout << "Dataset: #pts = " << npts << ", # dims = " << ndims << std::endl;

    size_t blk_size = 131072;
    size_t nblks = ROUND_UP(npts, blk_size) / blk_size;
    std::cout << "# blks: " << nblks << std::endl;
    
    std::ofstream writer(argv[2], std::ios::binary);
    
    // Write header: number of points and dimensions
    int npts_s32 = (int)npts;
    int ndims_s32 = (int)ndims;
    writer.write((char *)&npts_s32, sizeof(int));
    writer.write((char *)&ndims_s32, sizeof(int));
    
    // Allocate buffers for byte data
    uint8_t *read_buf = new uint8_t[blk_size * (ndims + sizeof(uint32_t))];
    uint8_t *write_buf = new uint8_t[blk_size * ndims];
    
    for (size_t i = 0; i < nblks; i++)
    {
        size_t cblk_size = std::min(npts - i * blk_size, blk_size);
        block_convert(reader, writer, read_buf, write_buf, cblk_size, ndims);
        std::cout << "Block #" << i << " written" << std::endl;
    }

    delete[] read_buf;
    delete[] write_buf;

    reader.close();
    writer.close();
    
    return 0;
}