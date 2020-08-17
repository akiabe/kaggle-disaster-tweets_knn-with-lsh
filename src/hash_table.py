import numpy as np

def hash_value_of_vector(v, planes):
    """create a hash for a vector
    :param v: vector of tweet, (1, N_DIMS)
    :param planes: the set of planes that divide up the region, (N_DIMS, N_PLANES)
    :return res: a number which is used as a hash for vector
    """
    # calculate the dot product between the vector and the matrix containing the planes
    dot_product = np.dot(v, planes)

    # get the sign of the dot product
    sign_of_dot_product = np.squeeze(np.sign(dot_product))

    # if the sign is negative, set h to be false, else set h to be true
    h = np.ones(dot_product.shape) * (sign_of_dot_product >= 0)

    # remove extra un-used dimensions and convert 2D to a 1D array
    h = np.squeeze(h)

    # initialize the hash value
    hash_value = 0

    n_planes = planes.shape[1]

    for i in range(n_planes):
        # increment the hash value by 2^i * h_i
        hash_value += 2 ** i * h[i]

    # cast hash_value as an integer
    hash_value = int(hash_value)

    return hash_value

def make_hash_table(vecs, planes):
    """
    :param vecs: list of vectors to be hashed
    :param planes: the matrix of planes in a single universe with shape (embedding dimensions, number of planes)
    :return hash_table: dictionary, keys are hashes, values are list of vectors (hash buckets)
    :return id_table: dictionary, keys are hashes, values are list of vectors id's
    """
    # number of planes is the number of columns in the planes matrix
    num_of_planes = planes.shape[1]

    # number of buckets is 2^(number of planes)
    num_buckets = 2 ** num_of_planes

    # create the hash table and id table as a dictionary
    hash_table = {i: [] for i in range(num_buckets)}
    id_table = {i: [] for i in range(num_buckets)}

    # for each vector in vecs
    for i, v in enumerate(vecs):
        # calculate the hash value for the vector
        h = hash_value_of_vector(v, planes)

        # store the vector into hash_table at key h
        hash_table[h].append(v)

        # store the vector's idx i into id_table at key h
        id_table[h].append(i)

    return hash_table, id_table

def approximate_knn(doc_id, v, planes_l, k=1, num_universe_to_use=N_UNIVERSES):
    """search for k-NN using hashes"""

    assert num_universe_to_use <= N_UNIVERSES

    # Vectors that will be checked as possible nearest neighbor
    vecs_to_consider_l = list()

    # list of document IDs
    ids_to_consider_l = list()

    # create a set for ids to consider, for faster checking if a document ID already exists in the set
    ids_to_consider_set = set()

    # loop through the universes of planes
    for universe_id in range(num_universes_to_use):

        # get the set of planes from the planes_l list, for this particular universe_id
        planes = planes_l[universe_id]

        # get the hash value of the vector for this set of planes
        hash_value = hash_value_of_vector(v, planes)

        # get the hash table for this particular universe_id
        hash_table = hash_tables[universe_id]

        # get the list of document vectors for this hash table, where the key is the hash_value
        document_vectors_l = hash_table[hash_value]

        # get the id_table for this particular universe_id
        id_table = id_tables[universe_id]

        # get the subset of documents to consider as nearest neighbors from this id_table dictionary
        new_ids_to_consider = id_table[hash_value]

        # remove the id of the document that searching
        if doc_id in new_ids_to_consider:
            new_ids_to_consider.remove(doc_id)
            print(f"removed doc_id {doc_id} of input vector from new_ids_to_search")

        # loop through the subset of document vectors to consider
        for i, new_id in enumerate(new_ids_to_consider):

            # if the document ID is not yet in the set ids_to_consider
            if new_id not in ids_to_consider_set:
                # access document_vectors_l list at index i to get the embedding
                # then append it to the list of vectors to consider as possible nearest neighbors
                document_vector_at_i = document_vectors_l[i]
                vecs_to_consider_l.append(document_vector_at_i)

                # append the new_id (the index for the document) to the list of ids to consider
                ids_to_consider_l.append(new_id)

                # also add the new_id to the set of ids to consider
                ids_to_consider_set.add(new_id)

    # run k-NN on the smaller set of vecs-to-consider.
    print("Fast considering %d vecs" % len(vecs_to_consider_l))

    # convert the vecs to consider set to a list, then to a numpy array
    vecs_to_consider_arr = np.array(vecs_to_consider_l)

    # call nearest neighbors on the reduced list of candidate vectors
    nearest_neighbor_idx_l = nearest_neighbor(v, vecs_to_consider_arr, k=k)

    # Use the nearest neighbor index list as indices into the ids to consider
    # create a list of nearest neighbors by the document ids
    nearest_neighbor_ids = [ids_to_consider_l[idx]
                            for idx in nearest_neighbor_idx_l]

    return nearest_neighbor_ids
































