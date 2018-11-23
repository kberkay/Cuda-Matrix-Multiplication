# Algorithm Description

Algorithm handles all matrices as square matrix. During research I have found that square matrices are multiplied in shorter times. For example multiplying 1024x1024 by 1024x1024 matrix takes 4 times less duration than 1024x1024 by 1024x1023 matrix, so I have transformed the matrices to square matrices by equalizing their dimension and filling empty places with zeros according to block size.

In the kernel because of the shared memory usage and its size limitations I have found solution from web [1] named “tiling”. By dividing the matrices to square tiles algorithm founds the one part of the resulting element and then considering other tiles and their result it finds one element of the resulting matrix. While using tiling solution and shared memory, there are two important things: Coalesced memory access and bank conflict. In order to prevent from uncoalesced access tiles are taken from global memory to shared memory row by row by as big as block size. When reaching to shared memory matrices elements corresponds to different banks which can be seen from code, so bank conflict is prevented in this way.

Another important point using shared memory is synchronization of threads. `__syncthreads()`  is used in order to fill shared memory before calculation start. If calculation phase starts before filling the shared memory threads will reach to empty places.

## Compilation & Execution

```bash
nvcc mult.cu  -o mult

./mult
Enter m n n k :
1024 1024 1024 1024
GPU time= 1.987008 ms
CPU time= 4085.987000 ms
Results are equal!

```

## Block Size

While considering block size checking GPU board’s specs is important. It supports 32 banks while reaching to shared memory so I have used 16 and 32 as block sizes. Different sizes are also tried in order to compare performances. Grid dimension is directly computed according to matrix dimensions.

## Test Results

Block Size of Tile | Matrix Dimension | GPU time in ms | CPU time in ms
--- | --- | :---:|:---:
8x8   | 16x16 & 16x16         |0.133    |0.0100
8x8   | 512x512 & 512x512     |0.295    |489.160
8x8   | 1024x1024 & 1024x1024 |2.299    |3896.926
16x16 | 16x16 & 16x16         |0.123    |0.0100
16x16 | 512x512 & 512x512     | 0.256 |465.544
16x16 | 1024x1024 & 1024x1024 | 1.966       |3885.956
32x32 | 16x16 & 16x16         | 0.143|0.0700
32x32 | 512x512 & 512x512     | 0.254       | 489.861
32x32 | 1024x1024 & 1024x1024 | 1.987|4085.987


## Discussion
Memory is not a problem when global memory is considered because 1024x1024 matrix needs 4MB of space. However, shared memory size is limited and in order to provide concurrent execution of blocks in one SM shared memory must be divided wisely.

## Environment
        Nvidia GeForce GTX850M
        Intel® Core™ i7-4700HQ CPU
        Cuda 10.0, V10.0.130

## References
[1]\
 <http://www.es.ele.tue.nl/~mwijtvliet/5KK73/?page=mmcuda&fbclid=IwAR0JgDDshTIpx-2Sv7goDK0TauD0Iz7DFeFtookp0loKFvV6jHcK8D7E7M8>

[2]\
<http://www.shodor.org/media/content/petascale/materials/UPModules/matrixMultiplication/moduleDocument.pdf>



## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

