import pickle
import numpy as np
import pandas as pd

import hash_table
import word_embeddings

if __name__ == "__main__":
    # load english word embeddings
    en_embeddings_subset = pickle.load(open("../input/en_embeddings.p", "rb"))

    # fetch train set
    df_train = pd.read_csv("../input/train.csv")
    train_x = df_train.text.values
    train_x = train_x.tolist()
    # train_y = df_train.target.values

    # get all document embeddings and store into dictionary
    document_vecs, ind2Tweet = word_embeddings.get_document_vecs(
        train_x,
        en_embeddings_subset
    )

    # number of vectors
    N_VECS = len(train_x)
    # vector dimensions
    N_DIMS = len(ind2Tweet[1])
    # number of planes
    N_PLANES = 9
    # number of times to repeat the hashing to improve the search
    N_UNIVERSES = 25

    # create the sets of planes
    np.random.seed(0)
    planes_l = [np.random.normal(size=(N_DIMS, N_PLANES))
                for _ in range(N_UNIVERSES)]

    # document_vecs, ind2Tweet
    doc_id = 0
    doc_to_search = train_x[doc_id]
    vec_to_search = word_embeddings.document_vecs[doc_id]

    nearest_neighbor_ids = hash_table.approximate_knn(
        doc_id,
        vec_to_search,
        planes_l,
        k=3,
        num_universe_to_use=5
    )

