import torch
import torch.distributed as dist
from contexttimer import Timer
import os

# benchmark the latency of async allreduce
# allreudce on two tensors list. There is no dependency between two lists.

class Benchmarker:
    def __init__(self, local_rank : int, tensor_size : int, step_num : int = 10, use_async_op : bool = False) -> None:
        self.comm_streams: torch.cuda.Stream = [torch.cuda.Stream() for i in range(2)]
        self.step_num = step_num
        self.local_rank = local_rank
        self.t1 = [torch.randn(tensor_size).cuda(local_rank) for i in range(self.step_num)]
        self.t2 = [torch.randn(tensor_size).cuda(local_rank) for i in range(self.step_num)]
        self.handlers = []
        self.curr_step = 0
        self.use_async_op = use_async_op

    def async_allreduce(self, t : torch.Tensor, stream : torch.cuda.Stream):
        if self.use_async_op:
            h = dist.all_reduce(t, async_op = True)
            self.handlers.append(h)
        else:
            with torch.cuda.stream(stream):
                dist.all_reduce(t)

    def run(self, t1_async : bool = False, t2_async : bool = False):
        """r
        issue two allreduce on two independent tensors.
        """
        if t1_async:
            self.async_allreduce(self.t1[self.curr_step], self.comm_streams[0])
        else:
            dist.all_reduce(self.t1[self.curr_step])
        if t2_async:
            self.async_allreduce(self.t2[self.curr_step], self.comm_streams[1])
        else:
            dist.all_reduce(self.t2[self.curr_step])
        self.curr_step += 1

    def finish(self):
        if self.use_async_op:
            for h in self.handlers:
                h.wait()
        else:
            for i in range(2):
                torch.cuda.current_stream().wait_stream(self.comm_streams[i])

    def benchmark(self, *args, **kwargs):
        if self.local_rank == 0:
            print(f"benchmarking {args}")
        self.curr_step = 0
        self.handlers = []
        torch.cuda.synchronize()
        with Timer() as T:
            for i in range(self.step_num):
                self.run(*args, **kwargs)
            self.finish()
            torch.cuda.synchronize()
        if self.local_rank == 0:
            print(f"rank {local_rank}: {self.step_num} steps elapsed {T.elapsed} sec")

if __name__ == '__main__':
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    dist.init_process_group("nccl", rank=local_rank, world_size=world_size)
    benchmarker = Benchmarker(local_rank, 1000000, 50, use_async_op = True)
    # warmup
    benchmarker.benchmark(False, False)

    benchmarker.benchmark(True, True)

    benchmarker.benchmark(False, True)

    benchmarker.benchmark(True, False)

    benchmarker.benchmark(False, False)
