#include "cpu-new-forward.h"

void conv_forward_cpu(float *output, const float *input, const float *mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
  /*
  Modify this function to implement the forward pass described in Chapter 16.
  The code in 16 is for a single image.
  We have added an additional dimension to the tensors to support an entire mini-batch
  The goal here is to be correct, not fast (this is the CPU implementation.)

  Function paramters:
  output - output
  input - input
  mask - convolution kernel
  Batch - batch_size (number of images in x)
  Map_out - number of output feature maps
  Channel - number of input feature maps
  Height - input height dimension
  Width - input width dimension
  K - kernel height and width (K x K)
  */

  const int Height_out = Height - K + 1;
  const int Width_out = Width - K + 1;

  // We have some nice #defs for you below to simplify indexing. Feel free to use them, or create your own.
  // An example use of these macros:
  // float a = in_4d(0,0,0,0)
  // out_4d(0,0,0,0) = a
  
  #define out_4d(i3, i2, i1, i0) output[(i3) * (Map_out * Height_out * Width_out) + (i2) * (Height_out * Width_out) + (i1) * (Width_out) + i0]
  #define in_4d(i3, i2, i1, i0) input[(i3) * (Channel * Height * Width) + (i2) * (Height * Width) + (i1) * (Width) + i0]
  #define mask_4d(i3, i2, i1, i0) mask[(i3) * (Channel * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

  // Insert your CPU convolution kernel code here
  for (int img_idx=0; img_idx<Batch; img_idx++)
  {
    for (int out_map_idx=0; out_map_idx < Map_out; out_map_idx++)
    {
      for (int out_h=0; out_h < Height_out; out_h++)
      {
        for (int out_w=0; out_w < Width_out; out_w++)
        {
          out_4d(img_idx, out_map_idx, out_h, out_w) = 0;
          // multiply mask with input
          for (int in_map_idx=0; in_map_idx < Channel; in_map_idx++)
          {
            for (int mask_h=0; mask_h < K; mask_h++)
            {
              for (int mask_w=0; mask_w < K; mask_w++)
              {
                int in_w = out_w + mask_w;
                int in_h = out_h + mask_h;
                out_4d(img_idx, out_map_idx, out_h, out_w) += in_4d(img_idx, in_map_idx, in_h, in_w ) * mask_4d(out_map_idx, in_map_idx, mask_h, mask_w);
              }
            }
          }
        }
      }
    }
  }

  #undef out_4d
  #undef in_4d
  #undef mask_4d

}