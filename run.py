import os
from condor import condor, Job, Configuration

conf = Configuration(request_CPUs=1, request_GPUs=1, gpu_memory_range=[8000,24000], cuda_capability=5.5)

with condor('condor') as sess:
    j = Job('python', 'reinforce.py', arguments=dict(env='CartPole-v1', max_episode=15000, tag='reinf'))
    sess.submit(j, conf)