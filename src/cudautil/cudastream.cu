cudaStream_t stream0; 

cudaStream_t getCudaStream() {
    if (!stream0) 
        cudaStreamCreate(&stream0);
    return stream0;
}