1. **Convolutional Layers**:
   - `Conv2dNormAct`: 2D convolution with optional normalization and activation
   - `ConvTranspose2dNormAct`: 2D transposed convolution with similar options

2. **ERB (Equivalent Rectangular Bandwidth) Operations**:
   - `erb_fb`: Function to create ERB filter banks

3. **Deep Filtering Operations**:
   - `DfOp`: A module for applying deep filtering operations on spectrograms

4. **Recurrent Neural Network Components**:
   - `GroupedGRULayer`: A grouped GRU (Gated Recurrent Unit) layer
   - `GroupedGRU`: A multi-layer grouped GRU
   - `SqueezedGRU`: A GRU with additional linear layers for input/output processing

5. **Utility Functions**:
   - `local_snr`: Calculates local Signal-to-Noise Ratio
   - `ExponentialUnitNorm`: Applies unit normalization to complex spectrograms

6. **Linear Layer Variants**:
   - `GroupedLinearEinsum`: A grouped linear layer using Einstein summation
   - `GroupedLinear`: A grouped linear layer using standard operations