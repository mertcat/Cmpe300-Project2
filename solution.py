import sys
import argparse
from mpi4py import MPI
from collections import defaultdict
import string

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# function which reads the actual text
def load_text(filepath):
    sentences = []

    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            stripped_line = line.strip()
            # adding non-empty line to the sentences list
            if stripped_line:
                sentences.append(stripped_line)

    return sentences


# making a set for vocabulary and stopword
def load_words(filepath):
    word_set = set()
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            word = line.strip()
            # adding the word (non-empty line) to the words list
            if word:
                word_set.add(word)
    return word_set


# lowercase function
def lowercase_chunk(chunk):
    return [s.lower() for s in chunk]


# the function which removes the punctuation
def remove_punctuation_chunk(chunk):
    punc_to_remove = string.punctuation     # storing the punctuations which should be removed
    processed = []
    for sentence in chunk:
        temp = []
        for char in sentence:
            if char not in punc_to_remove:  # determine whether the char is punctuation
                temp.append(char)

        no_punc = ''.join(temp)
        processed.append(no_punc)

    return processed


# the function which removes the given stopwords from the sentences
def remove_stopwords_chunk(chunk, stopwords):
    processed = []
    for sentence in chunk:
        words = sentence.split()
        s_no_stopwords = []
        for word in words:
            # check whether the current word is NOT in the stopwords set
            if word not in stopwords:   # if not in the stopwords list, adding to the sentence
                s_no_stopwords.append(word)
        processed.append(s_no_stopwords)
    return processed


# the function calculates the term frequency of the text
def compute_term_frequency(processed_sentences, vocabulary):
    partial_tf = {word: 0 for word in vocabulary}

    # iterating through every word in every sentence
    for sentence in processed_sentences:
        for word in sentence:
            if word in vocabulary:
                partial_tf[word] += 1

    return dict(partial_tf)


# MANAGER PATTERN 1
def manager_pattern_1(args):

    all_sentences = load_text(args.text)
    vocabulary = load_words(args.vocab)
    stopwords = load_words(args.stopwords)

    num_sentences = len(all_sentences)
    num_workers = size - 1  # total workers which are ranked 1 to size-1

    # calculating base size and remainder
    base_size = num_sentences // num_workers
    remainder = num_sentences % num_workers

    chunks = []
    start_index = 0
    # creating chunks for each worker
    for i in range(num_workers):
        chunk_size = base_size + (1 if i < remainder else 0)
        end_index = start_index + chunk_size

        # separating the sentences for the chunks
        chunk = all_sentences[start_index:end_index]
        chunks.append(chunk)
        start_index = end_index

    # sending sentences, vocab and stopwords data to the workers
    for worker_rank in range(1, size):
        data_to_send = {
            'sentences': chunks[worker_rank - 1],   # sending assigned sentences to the workers
            'vocabulary': vocabulary,
            'stopwords': stopwords
        }
        # TAG 1: to identift the initial data message
        comm.send(data_to_send, dest=worker_rank, tag=1)

    # creating a dictionary containing all vocab list
    final_results = defaultdict(int)

    # receiving the results of the term frequency from the workers
    for worker_rank in range(1, size):
        # TAG 2: to identify the return message which contains partial result
        partial_tf = comm.recv(source=worker_rank, tag=2)

        for word, count in partial_tf.items():
            final_results[word] += count    # incrementing the word counts at the result dict

    # sorting the results and printing them
    sorted_results = sorted(final_results.items())
    for word, count in sorted_results:
        print(f"{word}: {count}")


# WORKER PATTERN 1
def worker_pattern_1(args):
    # receiving the data from the manager
    data_received = comm.recv(source=0, tag=1)

    # unpack the data from manager
    sentences = data_received['sentences']
    vocabulary = data_received['vocabulary']
    stopwords = data_received['stopwords']

    processed_data = lowercase_chunk(sentences)                 # lowercasing
    processed_data = remove_punctuation_chunk(processed_data)   # punctuation removal
    processed_data = remove_stopwords_chunk(processed_data, stopwords)  # stopwords removal

    # partial result for the assigned chunk
    partial_tf = compute_term_frequency(processed_data, vocabulary)
    # sending the partial result to the manager
    comm.send(partial_tf, dest=0, tag=2)





# MANAGER PATTERN 2
def manager_pattern_2(args):
    all_sentences = load_text(args.text)
    num_sentences = len(all_sentences)
    chunk_div = 10  # can be adjusted between 5 and 20

    # calculating the chunk size
    if num_sentences // chunk_div > 1:
        chunk_size = num_sentences // chunk_div
    else:
        chunk_size = 1

    # creating chunks for worker 1
    chunks = [all_sentences[i:i + chunk_size]
              for i in range(0, num_sentences, chunk_size)]

    # all chunks go to worker 1
    for i, chunk in enumerate(chunks):
        # TAG 10: to send data chunks
        comm.send(chunk, dest=1, tag=10)

    # sending end-of-data signal to worker 1
    # TAG 11: end-of-data signal
    comm.send(None, dest=1, tag=11)

    # receiving the final result from worker 4
    # TAG 12: for final result from the worker 4
    final_results = comm.recv(source=4, tag=12)

    sorted_results = sorted(final_results.items())
    for word, count in sorted_results:
        print(f"{word}: {count}")


# WORKER PATTERN 2
def worker_pattern_2(args):
    worker_rank = rank

    PREV_RANK = rank - 1    # to determine source (manager or previous worker)
    NEXT_RANK = rank + 1    # to determine destination worker (or manager for worker 4)

    # loading the only necessary data for each worker
    # worker 3 needs stopwords
    if worker_rank == 3:
        stopwords = load_words(args.stopwords)

    # worker 4 needs vocabulary
    elif worker_rank == 4:
        vocabulary = load_words(args.vocab)

    # creating a result dictionary for worker 4
    if worker_rank == 4:
        final_result = {word: 0 for word in vocabulary}

    # main loop which receives the chunks, processes and sends to next node
    while True:
        # determining the communication source
        source_rank = 0 if worker_rank == 1 else PREV_RANK

        status = MPI.Status()
        chunk = comm.recv(source=source_rank, tag=MPI.ANY_TAG, status=status)

        # check whether the signal is TAG 11 (end-of-data)
        # if it is end, terminate the loop
        if status.Get_tag() == 11:
            break

        processed_chunk = None

        # worker 1 is specialized for lowercasing
        if worker_rank == 1:
            processed_chunk = lowercase_chunk(chunk)

        # worker 2 is specialized for punctuation removal
        elif worker_rank == 2:
            processed_chunk = remove_punctuation_chunk(chunk)

        # worker 3 is specialized for stopwords removal
        elif worker_rank == 3:
            processed_chunk = remove_stopwords_chunk(chunk, stopwords)

        # worker 4 is specialized for calculating the final result
        elif worker_rank == 4:
            partial_tf = compute_term_frequency(chunk, vocabulary)
            for word, count in partial_tf.items():
                final_result[word] += count
            continue
        # forwarding the data to the next rank except for worker 4
        if worker_rank < 4:
            comm.send(processed_chunk, dest=NEXT_RANK, tag=10)

    # finalization after end-of-data signal is received
    if worker_rank < 4:
        # sending EOD signal to the next worker
        comm.send(None, dest=NEXT_RANK, tag=11)
    elif worker_rank == 4:
        # sending final result to the manager
        comm.send(dict(final_result), dest=0, tag=12)


# Pattern 3 manager
def manager_pattern_3(args):
    all_sentences = load_text(args.text)
    #vocabulary = load_words(args.vocab)
    #stopwords = load_words(args.stopwords)

    num_workers = size - 1              # -1 to exclude manager.
    num_pipelines = num_workers // 4    # Pipelines have 4 workers.

    pipeline_partitions = []                        # Stores subset of sentences as a list.
    partitions_size = len(all_sentences)            # To distribute the sentences to the pipelines.
    remainder = len(all_sentences) % num_pipelines  # Hold the remainder sentences.

    start_index = 0     # As in manager_pattern_1

    for i in range (num_pipelines):
        # Calculate the size for the pipeline.
        # Add 1 if its one of the earlier pipelines, this also depends on wether remainder exists.
        current_partition_size = partitions_size + (1 if i < remainder else 0)

        # Create the partition itself by slicing the main list
        individual_partition = all_sentences[start_index : start_index + current_partition_size]
        pipeline_partitions.append(individual_partition)

        # Increment start_index to create the next pipeline.
        start_index += current_partition_size


    # Now feed the data to pipelines.
    # Define constants.
    CHUNK_SIZE = 10     # Between 5 and 20 as in the project directive.
    TAG_DATA = 10       # Constant for chunk size.
    TAG_DONE = 11       # To hold the end for chunk.

    for i in range(num_pipelines):
        partition = pipeline_partitions[i]
        
        # Pipeline 0 starts with worker 1 since manager occupies worker 0.
        # So rank = pipelineID * 4 + 1
        start_rank = (i * 4) + 1

        # Iterate through the pipeline incrementing by the CHUNK_SIZE
        for j in range(0, len(partition), CHUNK_SIZE):
            chunk = partition[j : j + CHUNK_SIZE]

            # Send this chunk to the worker
            comm.send(chunk, dest = start_rank, tag = TAG_DATA)

        # After sending all the text, send done signal
        comm.send(None, dest = start_rank, tag = TAG_DONE)


    # Now we collect the results
    final_term_frequency_results = defaultdict(int)
    TAG_RESULT = 12

    for i in range(num_pipelines):
        # Last worker holds the counts. Get this workers rank.
        last_rank = (i * 4) + 4

        # Pause until all messages are received.
        partial_term_frequency = comm.recv(source = last_rank, tag = TAG_RESULT)

        # Merge them into the final_term_frequency_results
        for word, count in partial_term_frequency.items():
            # Adds the word frequencies on top of each other to get the total number of occurences.
            final_term_frequency_results[word] += count

    sorted_results = sorted(final_term_frequency_results.items())
    for w, c in sorted_results:
        print(f"{w}: {c}")


def worker_pattern_3(args):
    worker_index = rank - 1             # Shift ranks to 0 based.
    stage_id = worker_index % 4         # Designates the pipeline stage 0 to 3, since this is a 4 stage pipeline.

    # Now get the upstream and downstream
    # Upstream -> preceding worker, downstream -> succeeding worker.
    # We use rank instead of stage_id since rank is the global variable.
    us_rank = rank - 1
    # If this is the first pipeline stage upstream is the manager. Handle this exception:
    if (stage_id == 0):
        us_rank = 0

    ds_rank = rank + 1
    # If this is the last pipeline stage downstream is manager.
    if (stage_id == 3):
        ds_rank = 0

    # Define constants.
    TAG_DATA = 10
    TAG_DONE = 11
    TAG_RESULT = 12     # These are as manager_pattern_3

    worker_stopwords = []
    worker_vocab = set()
    worker_term_frequency_counts = defaultdict(int)

    # Worker 3 (stage 2) needs the stopwords.
    if (stage_id == 2):
        worker_stopwords = load_words(args.stopwords)
    # Worker 4 needs vocabulary to count words
    elif (stage_id == 3):
        worker_vocab = load_words(args.vocab)


    # Implement processing
    while True:
        status = MPI.Status()

        # Wait until source_rank sends data
        data = comm.recv(source = us_rank, tag = MPI.ANY_TAG, status = status)
        # Get the tag
        tag = status.Get_tag()

        # Cover the case TAG_DONE
        if (tag == TAG_DONE):
            if (stage_id < 3):
                # If this is not the last stage pass the done signal to the next stage.
                comm.send(None, dest = ds_rank, tag = TAG_DONE)
            else:
                # If this is the last stage, send the final data to the manager
                comm.send(dict(worker_term_frequency_counts), dest = 0, tag = TAG_RESULT)
            break

        # If we are not finished do the data processing, NLP states:
        processed = None

        if (stage_id == 0):     # Lowercasing
            processed = lowercase_chunk(data)
        elif (stage_id == 1):   # Remove punctuation
            processed = remove_punctuation_chunk(data)
        elif (stage_id == 2):   # Remove stopwords
            processed = remove_stopwords_chunk(data, worker_stopwords)
        elif (stage_id == 3):   # Term freq. counting
            partial_term_frequency = compute_term_frequency(data, worker_vocab)

            # Add the partial term counts to our local total
            for w, c in partial_term_frequency.items():
                worker_term_frequency_counts[w] += c
            # We dont return anything for this stage of pipeline until the pipeline is finished.
            continue

        # Pass the data to the next stage in the pipeline. Stage 3 doesnt reach here.
        comm.send(processed, dest = ds_rank, tag = TAG_DATA)










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

    if args.pattern == 3:
        if rank == 0:
            manager_pattern_3(args)
        else:
            worker_pattern_3(args)



if __name__ == '__main__':
    main()