# Rotation-Optimized Multiplexed Parallel Convolution (RO-MPConv) and Parallel BSGS matrix-vector multiplication       
This is the supplementary implementation of 'Low-Latency Linear Transformations with Small Key Transmission for Private Neural Network on Homomorphic Encryption.'       

Our implementation is based on [Lattigo v5](https://github.com/tuneinsight/lattigo/tree/v5.0.2), which is written in Go.

To run this project, please ensure that **Go (version 1.18 or higher)** is installed on your system.

Since we use Lattigo library to run the code, our implementation's location is fixed at `examples/rotopt/`.     

---

## Run
You can run RO-MPConv latency test as follows:     
```bash
cd examples/rotopt/   
go run . conv      
```    

Alternatively, you can specify a test function using arguments:   
```bash
go run . matVecMul conv          
```    
---

## Arguments
|Argument|Description|Related Figure/Table|
|---|---|---|
|`basic`|Execution time of RNS-CKKS basic operations (rotation, multiplication, addition).|Fig. 1|
|`conv`|Latency comparison for ResNet convolutions (CONV1, CONV2, CONV3s2, etc.).|Fig. 13|
|`otherConv`|Latency comparison for modern architecture convolutions (CvT, MUSE).|Fig. 13|
|`applications`|Convolution latency improvements for MPCNN, AutoFHE, CryptoFace, etc.|Table 4|
|`blueprint`|Extracts the blueprints for each RO-MPConv configuration.|Appendix A|
|`fc`|Performance of parallel BSGS mat-vec mul on a fully connected layer scenario.|Fig. 15|
|`matVecMul`|General performance comparison of BSGS matrix-vector multiplication methods.|Fig. 15|
|`ALL`|If you write ALL or don't write any args, all of the test functions will be started.|-|

---

## Algorithm Structure

The core algorithms are located in the `examples/rotopt/modules` directory.
- `convConfig.go` defines the convolution configurations corresponding to Appendix A and B of the paper.
---


## Notes

- The convolution operations for `otherConv` include:
  - **CvTConv1**: Used in Stage 2 of the Convolutional Vision Transformer (CvT) on CIFAR-100.
  - **CvTConv2**: Used in Stage 3 of CvT on CIFAR-100.
  - **MuseConv**: Used to generate a multi-scale feature pyramid from CLIP in the MUSE model.


