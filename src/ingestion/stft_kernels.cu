#include <cuda_runtime.h>
#include <math.h>

// Define Pi for the Hann Window calculation
#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif

extern "C" {

    /**
     *  Applies a pre-emphasis filter to the raw audio waveform.
     * Formula: y(t) = x(t) - alpha * x(t-1)
     */
    __global__ void pre_emphasis_kernel(const float* input, float* output, int length, float coeff) {
        // Calculate the global thread ID
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        
        if (idx < length) {
            if (idx == 0) {
                // The first sample has no previous sample to subtract
                output[idx] = input[idx];
            } else {
                output[idx] = input[idx] - coeff * input[idx - 1];
            }
        }
    }

    /**
     * u/brief Applies a Hann Window to an audio frame prior to FFT.
     * This prevents spectral leakage at the edges of the audio chunks.
     */
    __global__ void apply_hann_window_kernel(float* frame, int frame_length) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        
        if (idx < frame_length) {
            // Hann window formula: 0.5 * (1 - cos(2*pi*n / (N-1)))
            float multiplier = 0.5f * (1.0f - cosf((2.0f * M_PI * idx) / (frame_length - 1.0f)));
            frame[idx] = frame[idx] * multiplier;
        }
    }

} // extern "C"