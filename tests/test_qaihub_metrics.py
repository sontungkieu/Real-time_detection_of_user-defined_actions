from __future__ import annotations

from activity_recognition.utils.qaihub_metrics import (
    parse_runtime_analysis_text,
    parse_runtime_log,
)


def test_parse_runtime_log_extracts_timing_memory_and_fallback(tmp_path) -> None:
    log_path = tmp_path / "runtime.log"
    log_path.write_text(
        "\n".join(
            [
                "[21/May/2026:09:23:19 +00:00: profiler/info] Android system property: ro.product.model = SM-S926U1",
                "[21/May/2026:09:23:19 +00:00: profiler/info] OpenCL Version: OpenCL C 3.0 Adreno(TM) 750",
                "[21/May/2026:09:23:19 +00:00: profiler/info] [job_id: jp3qljll5] [model.tflite] No delegates specified; using compute unit=cpu_and_npu.",
                "[21/May/2026:09:23:19 +00:00: profiler/info] [job_id: jp3qljll5] [model.tflite] Loaded QNN Delegate, API version=0.24.0, QNN version=v2.45.0, capabilities: DSP=false, HTP/int8=true.",
                "[21/May/2026:09:23:20 +00:00: profiler/info] [job_id: jp3qljll5] [model.tflite] [tflite] Replacing 17 out of 21 node(s) with delegate (TfLiteQnnDelegate) node, yielding 9 partitions for subgraph 0.",
                "[21/May/2026:09:23:20 +00:00: profiler/info] [job_id: jp3qljll5] [model.tflite] Applied 1 delegates: QNN/HTP. Model is fully delegated=false.",
                "[21/May/2026:09:23:20 +00:00: profiler/info] [job_id: jp3qljll5] [model.tflite] Status Successfully Loaded Cold with t = 322381 us and usage: before = 93172.0 kB; peakBefore = 93172.0 kB; mallocUnusedBefore = 7191.9 kB; after = 121384.0 kB; peakAfter = 118036.0 kB; mallocUnusedAfter = 7683.4 kB; increase = 20528.6-27720.5 kB; peak = 24864.0-32055.9 kB",
                "[21/May/2026:09:23:21 +00:00: profiler/info] -=- Tungsten running task: performing inference -=-",
                "[21/May/2026:09:23:21 +00:00: profiler/info] [job_id: jp3qljll5] [model.tflite] Successfully ran model for 100 iterations across 1 batches in 0.034 sec.",
                "[21/May/2026:09:23:21 +00:00: profiler/info] [job_id: jp3qljll5] [model.tflite] Status Successfully Performed Inference with t = 300 us and usage: before = 116128.0 kB; peakBefore = 116128.0 kB; mallocUnusedBefore = 25893.0 kB; after = 116208.0 kB; peakAfter = 116204.0 kB; mallocUnusedAfter = 25884.5 kB; increase = 0.0-88.5 kB; peak = 76.0-25969.0 kB",
                "[21/May/2026:09:23:22 +00:00: profiler/info] -=- Tungsten running task: performing inference by layer -=-",
                "[21/May/2026:09:23:22 +00:00: profiler/debug] [job_id: jp3qljll5] [model.tflite] Populating InferByLayer results - node=:0:25, tag=TfLiteQnnDelegate, time=191us, cycles=0.",
                "[21/May/2026:09:23:22 +00:00: profiler/debug] [job_id: jp3qljll5] [model.tflite] Populating InferByLayer results - node=:0:15, tag=BATCH_TO_SPACE_ND, time=0us, cycles=0.",
                "[21/May/2026:09:23:22 +00:00: profiler/info] [job_id: jp3qljll5] [model.tflite] Status Successfully Performed Inference By Layer with t = 854 us and usage: before = 117672.0 kB; peakBefore = 117672.0 kB; mallocUnusedBefore = 25925.5 kB; after = 117748.0 kB; peakAfter = 117720.0 kB; mallocUnusedAfter = 25898.8 kB; increase = 0.0-102.7 kB; peak = 48.0-25973.5 kB",
            ]
        ),
        encoding="utf-8",
    )

    record = parse_runtime_log(log_path)

    assert record["job_id"] == "jp3qljll5"
    assert record["compute_unit_observed"] == "cpu_and_npu"
    assert record["delegate"]["delegated_nodes"] == 17
    assert record["delegate"]["total_nodes"] == 21
    assert record["delegate"]["fully_delegated"] is False
    assert record["timings"]["inference_us"] == 300
    assert record["timings"]["inference_wall_time_100_iter_sec"] == 0.034
    assert record["memory"]["inference"]["increase_max_mb"] == 0.086426
    assert record["by_layer"]["summary"]["fallback_entries"] == 1


def test_parse_runtime_analysis_copy_paste_table() -> None:
    text = """
Layer
Type
Kernel(s)
Placement
Compute Cycles
Timing
tinytcn_1/conv
CONV_2D
Transpose
Conv2d
Relu
NPU (QNN)
9,109
5 μs
tinytcn_1/convolution/SpaceToBatchND
SPACE_TO_BATCH_ND
SPACE_TO_BATCH_ND
CPU (TfLite)
0 μs
"""

    analysis = parse_runtime_analysis_text(text, job_id="jp3qljll5")

    assert analysis["job_id"] == "jp3qljll5"
    assert analysis["summary"]["total_entries"] == 2
    assert analysis["summary"]["fallback_entries"] == 1
    assert analysis["summary"]["placement_counts"]["NPU (QNN)"] == 1
    assert analysis["summary"]["placement_cycles"]["NPU (QNN)"] == 9109


def test_parse_runtime_analysis_tsv_table() -> None:
    text = "\n".join(
        [
            "Layer\tType\tKernel(s)\tPlacement\tCompute Cycles\tTiming",
            "layer0\tCONV_2D\tTranspose;Conv2d\tNPU (QNN)\t9,109\t5 μs",
            "layer1\tSPACE_TO_BATCH_ND\tSPACE_TO_BATCH_ND\tCPU (TfLite)\t\t0 μs",
        ]
    )

    analysis = parse_runtime_analysis_text(text, job_id="job")

    assert analysis["summary"]["total_entries"] == 2
    assert analysis["summary"]["fallback_ops"]["SPACE_TO_BATCH_ND"] == 1
    assert analysis["summary"]["total_cycles"] == 9109
