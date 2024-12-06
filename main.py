with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA],
    on_trace_ready=torch.profiler.tensorboard_trace_handler('./log'),
    record_shapes=True,
    with_stack=True
) as prof:
    fit_one_cycle(modelNet, train_loader_gpu, test_loader_gpu, 0, num_epochs, deviceGPU)
print(prof.key_averages().table(sort_by="cuda_time_total"))