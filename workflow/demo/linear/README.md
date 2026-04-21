# Linear Workflow Demo

Run these commands from the repository root.

The linear demo uses the dedicated entrypoint in `workflow/demo/linear/main.py`:

```bash
python -m workflow.demo.linear.main --help
```

## Requirements

- `--model_path` is required.
- `--vllm_sparse_method` is optional and only needed when you want to enable a specific Sparse-vLLM method.
- `--hyper_params` accepts a JSON object and is passed through to the Sparse-vLLM runtime.

## Mode 1: Single

Use `single` mode to run one linear workflow from end to end:

```bash
python -m workflow.demo.linear.main \
  --mode single \
  --model_path /path/to/model \
  --question "What is the best rollout plan for this feature?" \
  --context "The feature adds a new workflow runtime. The team wants a staged rollout, strong observability, and explicit risk handling before production traffic is raised." \
  --print_history
```

What this does:

- Builds the 3-step linear workflow: `researcher -> reasoner -> extractor`
- Starts one workflow execution
- Prints the final workflow state
- Optionally prints per-node history, including the materialized prompt sent to the LLM engine, when `--print_history` is set

Useful flags for `single` mode:

- `--researcher_max_tokens`, `--reasoner_max_tokens`, `--extractor_max_tokens`
- `--temperature`
- `--max_inflight_requests`
- `--max_model_len`

## Mode 2: Traffic

```bash
bash run_linear_traffic_vanilla_longctx.sh
```

```text
============================================================================================================================================
Method       Len      BS   TTFT(s)    PreTP        DecTP        ITL(ms)    AvgBS    Mem(GB)    Speedup
--------------------------------------------------------------------------------------------------------------------------------------------
vanilla      131398   16*  196.54     3432.6       122.5        130.57     6.6      73.07      1.00x
============================================================================================================================================
```

```bash
bash run_linear_traffic_snapkv_longctx.sh
```

```text
============================================================================================================================================
Method       Len      BS   TTFT(s)    PreTP        DecTP        ITL(ms)    AvgBS    Mem(GB)    Speedup
--------------------------------------------------------------------------------------------------------------------------------------------
snapkv       131398   16*  203.01     3361.3       443.2        36.10      15.3     77.14      -
============================================================================================================================================
```
