import sys
import argparse
from mpi4py import MPI
from collections import defaultdict
import string

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def load_text(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        text = f.read()
    sentences = []

    text = text.replace('!', '.').replace('?', '.').replace(',', '.')
    for sentence in text.split('.'):
        clean_s = sentence.strip()
        if clean_s:
            sentences.append(clean_s)
    return sentences


def load_words(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return {word.strip().lower() for line in f for word in line.split() if word.strip()}


def lowercase_chunk(chunk):
    return [s.lower() for s in chunk]


def remove_punctuation_chunk(chunk):
    punc_to_remove = string.punctuation
    processed = []
    for sentence in chunk:
        temp = []
        for char in sentence:
            if char not in punc_to_remove:
                temp.append(char)

        no_punc = ''.join(temp)
        processed.append(no_punc)

    return processed



def remove_stopwords_chunk(chunk, stopwords):
    processed = []
    for sentence in chunk:
        words = sentence.split()
        s_no_stopwords = [word for word in words if word not in stopwords]
        processed.append(s_no_stopwords)
    return processed


def compute_term_frequency(processed_sentences, vocabulary):
    partial_tf = defaultdict(int)

    # Iterate through every word in every sentence
    for sentence in processed_sentences:
        for word in sentence:
            if word in vocabulary:
                partial_tf[word] += 1

    return dict(partial_tf)


def manager_pattern_1(args):

    all_sentences = load_text(args.text)
    vocabulary = load_words(args.vocab)
    stopwords = load_words(args.stopwords)

    num_sentences = len(all_sentences)
    num_workers = size - 1  # Ranks 1 to size-1

    if num_workers == 0:
        return

    # Calculate base size and remainder
    base_size = num_sentences // num_workers
    remainder = num_sentences % num_workers

    chunks = []
    start_index = 0
    for i in range(num_workers):
        chunk_size = base_size + (1 if i < remainder else 0)
        end_index = start_index + chunk_size

        chunk = all_sentences[start_index:end_index]
        chunks.append(chunk)
        start_index = end_index

    for worker_rank in range(1, size):
        data_to_send = {
            'sentences': chunks[worker_rank - 1],
            'vocabulary': vocabulary,
            'stopwords': stopwords
        }
        comm.send(data_to_send, dest=worker_rank, tag=1)

    final_tf_results = defaultdict(int)

    for worker_rank in range(1, size):
        partial_tf = comm.recv(source=worker_rank, tag=2)

        for word, count in partial_tf.items():
            final_tf_results[word] += count

    sorted_results = sorted(final_tf_results.items())
    for word, count in sorted_results:
        print(f"{word}: {count}")


def worker_pattern_1(args):

    data_received = comm.recv(source=0, tag=1)

    sentences = data_received['sentences']
    vocabulary = data_received['vocabulary']
    stopwords = data_received['stopwords']

    processed_data = lowercase_chunk(sentences)
    processed_data = remove_punctuation_chunk(processed_data)
    processed_data = remove_stopwords_chunk(processed_data, stopwords)

    partial_tf = compute_term_frequency(processed_data, vocabulary)

    comm.send(partial_tf, dest=0, tag=2)






def manager_pattern_2(args):
    all_sentences = load_text(args.text)

    num_sentences = len(all_sentences)

    CHUNK_DIVISOR = 10  # Can be adjusted between 5 and 20

    if num_sentences // CHUNK_DIVISOR > 1:
        chunk_size = num_sentences // CHUNK_DIVISOR
    else:
        chunk_size = 1


    chunks = [all_sentences[i:i + chunk_size]
              for i in range(0, num_sentences, chunk_size)]

    worker_1_rank = 1
    TAG_DATA = 10
    TAG_EOD = 11

    for i, chunk in enumerate(chunks):
        comm.send(chunk, dest=worker_1_rank, tag=TAG_DATA)

    comm.send(None, dest=worker_1_rank, tag=TAG_EOD)

    worker_4_rank = 4
    TAG_RESULT = 12

    final_tf_results = comm.recv(source=worker_4_rank, tag=TAG_RESULT)

    sorted_results = sorted(final_tf_results.items())
    for word, count in sorted_results:
        print(f"{word}: {count}")


def worker_pattern_2(args):
    worker_rank = rank


    PREV_RANK = rank - 1
    NEXT_RANK = rank + 1

    TAG_DATA = 10
    TAG_EOD = 11
    TAG_RESULT = 12

    if worker_rank == 3:
        stopwords = load_words(args.stopwords)

    elif worker_rank == 4:
        vocabulary = load_words(args.vocab)

    if worker_rank == 4:
        final_tf = defaultdict(int)

    while True:
        source_rank = 0 if worker_rank == 1 else PREV_RANK

        status = MPI.Status()
        chunk = comm.recv(source=source_rank, tag=MPI.ANY_TAG, status=status)


        if status.Get_tag() == TAG_EOD:
            break

        processed_chunk = None

        if worker_rank == 1:
            processed_chunk = lowercase_chunk(chunk)
        elif worker_rank == 2:
            processed_chunk = remove_punctuation_chunk(chunk)
        elif worker_rank == 3:
            processed_chunk = remove_stopwords_chunk(chunk, stopwords)
        elif worker_rank == 4:
            partial_tf = compute_term_frequency(chunk, vocabulary)
            for word, count in partial_tf.items():
                final_tf[word] += count
            continue

        if worker_rank < 4:
            comm.send(processed_chunk, dest=NEXT_RANK, tag=TAG_DATA)

    if worker_rank < 4:
        comm.send(None, dest=NEXT_RANK, tag=TAG_EOD)
    elif worker_rank == 4:
        comm.send(dict(final_tf), dest=0, tag=TAG_RESULT)















def main():

    if rank == 0:
        parser = argparse.ArgumentParser(description="CMPE 300 Project 2 - MPI-Based Parallel NLP System")
        parser.add_argument('--text', type=str, required=True, help='Path to the input text file.')
        parser.add_argument('--vocab', type=str, required=True, help='Path to the vocabulary file.')
        parser.add_argument('--stopwords', type=str, required=True, help='Path to the stopwords file.')
        parser.add_argument('--pattern', type=int, required=True, choices=[1, 2, 3, 4],
                            help='Indicates which processing pattern to execute.')


        args = parser.parse_args()

        if args.pattern == 1 and size < 2:
            sys.exit(1)
        comm.bcast(args, root=0)

    else:
        args = comm.bcast(None, root=0)

    if args.pattern == 1:
        if rank == 0:
            manager_pattern_1(args)
        else:
            worker_pattern_1(args)

    if args.pattern == 2:
        if rank == 0:
            manager_pattern_2(args)
        else:
            worker_pattern_2(args)



if __name__ == '__main__':
    main()