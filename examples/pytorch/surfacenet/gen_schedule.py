import argparse

parser = argparse.ArgumentParser(description='Schedule Generator')
parser.add_argument('--use-threshold', action='store_true')
parser.add_argument('--threshold-dgl', type=int, default=None)
parser.add_argument('--threshold-padding', type=int, default=None)
parser.add_argument('--batch-size', type=int, default=8)
parser.add_argument('--max-vertices', type=int, default=100000)
args = parser.parse_args()

MAX_INT = 10000000000
if args.threshold_dgl is None:
    args.threshold_dgl = MAX_INT
if args.threshold_padding is None:
    args.threshold_padding = MAX_INT

with open("vertices.txt") as f:
    nodes = [int(line.strip()) for line in f]

def loader():
    sample_id = 0
    num_samples = len(nodes)
    batch_count = 0
    samples = []
    max_vertices = 0
    batch_size = 0
    total_vertices = 0

    while sample_id < num_samples:
        new_sample = sample_id
        sample_id += 1
        num_vertices = nodes[new_sample]
        if num_vertices > args.max_vertices:
            # skip graphs that are larger than threshold
            continue

        if args.use_threshold and (max(max_vertices, num_vertices) * (batch_size + 1) > args.threshold_padding or total_vertices + num_vertices > args.threshold_dgl):
            yield samples
            samples = []
            batch_count += 1
            batch_size = 0
            max_vertices = 0
            total_vertices = 0

        max_vertices = max(max_vertices, num_vertices)
        batch_size += 1
        samples.append(new_sample)
        total_vertices += num_vertices

        if not args.use_threshold and batch_size == args.batch_size:
            yield samples
            samples = []
            batch_count += 1
            batch_size = 0
            max_vertices = 0
            total_vertices = 0

    if len(samples) > 0:
        yield samples

for samples in loader():
    num_vertices = [nodes[i] for i in samples]
    size_after_padding = max(num_vertices) * len(samples)
    if size_after_padding <= args.threshold_padding and sum(num_vertices) <= args.threshold_dgl:
        print(",".join(map(str, samples)))
    else:
        print(size_after_padding, sum(num_vertices))
        assert 0
