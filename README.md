This is a pytorch implementation of SimNet.

Instead of using the original definition of SimNet, I did some change:
- drop the bias of MEX to Similarity to save time
- replace patch wise bias to channel-wise bias, which is the same to the original convolution
- replace the R+ constrain of u in Similarity to R. (the original code looks like replace too!)


Note that I use im2col style conv in Similarity, which is slower than the FFT style conv in CUDNN. There are no good method with this problem now.

TODO:
- cifar10, replace the conv to Similarity, the Maxpooling to Mex
