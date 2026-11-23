# Comparison

Figures of other designs are taken from their papers (listed at the end). Ours are from `results/README.md`: 250 MHz,
post-route out of context, power from the Vivado estimate.

## 1. Mamba accelerators

| Design | Workload | Device (nm) | MHz | Arithmetic | kLUT | DSP | BRAM / URAM | Throughput | GOP/s <sup>j</sup> | Power | GOP/s/W |
|---|---|---|---|---|---|---|---|---|---|---|---|
| MARCA [1] | Mamba1 130M-2.8B, LM | ASIC (28) | 1,000 | 32-bit fixed | 221.9 mm² (synthesized) | | | speedups only | - | 10.44 W | - |
| FastMamba [2] | Mamba2-2.7B, decode | VC709 (28) | 250 | W8A8, PoT SSM | 334.8 | 3,333 | 956 / - | 5.68 tok/s | 30.7 | 9.3 W <sup>a</sup> | 3.3 |
| LightMamba [3] | Mamba2-2.7B, decode | VCK190 (7) | 400 | W4A4 | 107 | 228 | 912 / 61 | 7.21 tok/s | 38.9 | 3.2 W <sup>a</sup> | 12.3 |
| SpecMamba [4] | Mamba2-2.7B <sup>b</sup>, decode | VCK190 (7) | 400 | INT4 | 536 | 1,106 | 1,223 / 141 | 20.6 tok/s | 111 | - <sup>b</sup> | - |
| LowRank-SSM [5] | Mamba2-2.7B <sup>c</sup>, decode | VC1902 (7) | 230 | INT8 weights | 233.5 | 1,624 | 309.5 / 101 | 16.79 tok/s | 90.7 | 6.9 W <sup>d</sup> | 13.2 |
| eMamba [6] | Mamba, point clouds | ZCU102 (16) | 100 | INT8 | 188 <sup>e</sup> | ≈297 <sup>e</sup> | 0 / - | 16.4 µs/frame | - | - | - |
| ViM-Q [7] | ViM-t, images | ZCU102 (16) | 350 | W4A8 | 121.7 <sup>f</sup> | 1,377 | 737 / - | 53.8 ms/image | - | 12.8 W <sup>g</sup> | - |
| **This work** | **ITMN-Mamba2, ECG** | **KV260 (16)** | **250** | **16-bit fixed** | **56.1** | **146** | **116 / 32** | **9.90 ms/record** | **45.2** | **2.03 W <sup>g</sup>** | **22.3** |

Normalized to the resources (designs that report a throughput in operations):

| Design | GOP/s per DSP | GOP/s per kLUT |
|---|---|---|
| FastMamba | 0.009 | 0.092 |
| LightMamba | 0.171 | 0.364 |
| SpecMamba | 0.100 | 0.207 |
| LowRank-SSM | 0.056 | 0.388 |
| **This work** | **0.310** | **0.806** |

Mamba1, Mamba2: the two generations of the Mamba block. LM: language model.
<sup>a</sup> Derived from the reported energy efficiency (LightMamba: measured on board).
<sup>b</sup> With a 370M draft model (speculative decoding); energy efficiency reported only relative to LightMamba (1.26x).
<sup>c</sup> Low-rank projections.
<sup>d</sup> Board power minus idle power.
<sup>e</sup> From the reported 68.5 % LUT and 11.8 % DSP utilization of the ZCU102.
<sup>f</sup> Plus 27.5 k LUTs used as memory.
<sup>g</sup> Vivado estimate.
<sup>j</sup> For the language models, dense-equivalent: two operations per model parameter per generated token
(5.4 GOP for Mamba2-2.7B) times the reported tokens per second. Batch-1 decoding is memory-bound, and LowRank-SSM's
low-rank projections perform fewer operations, so these figures measure delivered model work, not peak compute.
Ours: 0.447 GOP per record (223.5 M MAC, the scan included) / 9.90 ms.

Reading the table:
- The language-model designs decode Mamba2-2.7B with weights streamed from DRAM in 4- or 8-bit arithmetic; this
  design runs a 0.4 M-parameter model from on-chip memory in 16-bit arithmetic. Throughput and per-resource figures
  measure delivered work in each design's own setting.
- Our power is an estimate of the accelerator alone; the measured figures of [2], [3] and [5] include the board's
  processing system and DRAM.
- The vision designs [6], [7] report no operation counts; they use 2.2 to 3.4x our LUTs and 2.0 to 9.4x our DSPs.

### Projection to language-model size

From a cycle model of the mixer (stage costs as functions of the dimensions, fitted to the 14 measured mixer
configurations, which it reproduces within 0.9 %), with d_state 128 and headdim 64, decoding one token through all
layers including the LM head:

| Model | Compute per token | Bound by memory traffic (16-bit weights and state per token) |
|---|---|---|
| Mamba2-130M | 4.74 ms (211 tok/s) | ≤ 69 tok/s at the KV260's 19.2 GB/s |
| Mamba2-2.7B | 90.5 ms (11.1 tok/s) | ≤ 2.2 tok/s at 12 GB/s (5.4 GB of weights exceed the KV260's 4 GB) |

These assume a weight path that streams into double-buffered weight banks while the datapath computes. The design as
built loads weights only while idle, through its 256-bit host port. At this scale the datapath would be limited by
memory traffic, not by its 146 DSPs; 8- and 4-bit weight formats as in [2], [3], [5] would cut that traffic.

## 2. FPGA ECG classifiers

| | MINA [8] | ECG-LDC [9] | This work |
|---|---|---|---|
| Data | MIT-BIH | MIT-BIH | PTB-XL |
| Input | one beat | one beat | 12 leads, 10 s |
| Task | 5 beat classes | 5 beat classes | 5 diagnostic classes, multi-label |
| Model | 1-D CNN | hypervectors | Inception + Mamba2 |
| Parameters | 6.5 k | 3.86 kB model | 398 k |
| Platform | ZCU102 | Pynq-Z2 <sup>a</sup> | KV260 |
| MHz | 250 | 50 | 250 |
| Precision | 16-bit fixed | binary, 8-bit input | 16-bit fixed |
| LUT / FF | 24.7 k / 11.2 k | 5.7 k / 4.8 k | 56.1 k / 37.9 k |
| BRAM / DSP | 4.5 / 40 | 4 / 0 | 116 (+32 URAM) / 146 |
| Latency | 34.45 µs/beat | 23.9 µs/beat | 9.90 ms/record |
| GOP/s | 9.85 | - | 45.2 |
| GOP/s per DSP | 0.246 | - | 0.310 |
| eLUT = LUT + 280 DSP + 784 BRAM | 39.4 k | 8.8 k | 188 k <sup>c</sup> |
| GOP/s per M eLUT | 249.9 | - | 241 <sup>c</sup> |
| Power | 0.83 W | 0.14 W <sup>b</sup> | 2.03 W <sup>b</sup> |
| Accuracy | 99.37 % acc. | 97.18 % acc. | 0.919 macro AUC |

<sup>a</sup> Accelerator core. With the processor and the AXI subsystem: 8.3 k LUTs, 4.20 ms per beat, 1.71 W.
<sup>b</sup> Vivado estimate for the accelerator.
<sup>c</sup> UltraRAM excluded (the metric does not define it); counting each URAM as eight BRAMs: 389 k eLUT,
116 GOP/s per M eLUT.

The ECG designs classify single heartbeats with small models. This design classifies a whole 12-lead, 10-s record
into five overlapping diagnostic classes with a 60x larger model: 0.45 GOP per record against 0.33 MOP per beat for
MINA, at 4.6x MINA's throughput and the same clock. On MINA's area-normalized metric it reaches 241 against MINA's
250; the 2-D CNN ECG accelerators compared in [8] reach 66 to 150.

## 3. Same model on CPU and GPU

Batch 1, float model (PyTorch, `mamba_ssm` kernels on the GPU, the library's chunked reference on the CPU; their
logits agree to 7e-4). Medians over 200 (GPU) and 50 (CPU) runs.

| Platform | Format | ms / record | W | mJ / record |
|---|---|---|---|---|
| Core i5-12500H, 8 threads | FP32 | 20.3 | - | - |
| GeForce RTX 3050 Laptop | FP32 | 10.1 | 32.3 (board) | 326 |
| **This work, KV260** | **Q4.11** | **9.90** | **2.03 (on-chip estimate)** | **20.0** |

The accelerator matches the GPU's single-record latency at 16x less energy per record (its figure excludes the KV260's
processing system and DRAM) and is 2x faster than the CPU. With batches of 64 the GPU reaches 0.65 ms per record,
which suits offline screening of stored recordings; the accelerator targets one record at a time next to the sensor.
A record is 10 s of signal, so the accelerator runs about 1,000x faster than real time.

## References

1. "MARCA: Mamba accelerator with reconfigurable architecture," ICCAD 2024.
2. "FastMamba: A high-speed and efficient Mamba accelerator on FPGA with accurate quantization," arXiv:2505.18975.
3. "LightMamba: Efficient Mamba acceleration on FPGA with quantization and hardware co-design," arXiv:2502.15260.
4. "SpecMamba: Accelerating Mamba inference on FPGA with speculative decoding," ICCAD 2025.
5. "LowRank-SSM: Hardware-software co-design for rank-reduced Mamba acceleration on FPGA," arXiv:2608.02954.
6. "eMamba: Efficient acceleration framework for Mamba models in edge computing," ACM TECS 2025.
7. "ViM-Q: Scalable algorithm-hardware co-design for vision Mamba model inference on FPGA," arXiv:2605.01935.
8. "MINA: A hardware-efficient and flexible mini-InceptionNet accelerator for ECG classification in wearable
   devices," IEEE TCAS-I 2025.
9. "ECG-LDC: A hardware-efficient low-dimensional computing framework for ECG arrhythmia classification,"
   arXiv:2607.09680.
