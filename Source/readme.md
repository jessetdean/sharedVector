<h1 align="center">SharedVector (Cuda/Host)</h1>

<h1>How to Install</h1>

Copy all files in the "sharedVector" folder into your project directory. Include only "sharedVector.h" in host code and "cudaVector.h" in .cu files.

<h1>How to Use</h1>

The idea of this project is to have one set of code for datastructures that can be used either in host or device code. Switch from host to device by simply changing a boolean argument during the render step. The rendered cudaVectors can be used directly in a kernel parameter list.

SharedVector is an extension of the STL vector class and can be used in an identical mannner. Do pushes, resizes and similar memory-altering operations on this object.

Before using in the main part of code (that can/should be __host__ __device__ function calls), render the sharedVector to cudaVector (or cuClass to its base object). To get data back, use a sync() call. Note: the sharedVector should not change size before sync() is called.

Included in this repo is an example (Dataset) of how to use these functions effectively.