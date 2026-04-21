import pickle
import os, sys
from loguru import logger
from tqdm import tqdm
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_dir)
from common.common_utils import read_json, read_jsonl, save_to_pkl
from tool.client_freebase import DATATYPE, TYPE, FreebaseClient, add_not_exists_or_exists_filter
from tool.openai import get_embedding_batch, get_embedding

def get_type():
    fb = FreebaseClient(end_point="http://localhost:8890/sparql")

    spq_query = """
    PREFIX ns: <http://rdf.freebase.com/ns/>
    SELECT DISTINCT ?type
    WHERE {
        ?e ns:type.object.type ?type
    }
    """
    query_types = fb.query(spq_query)
    types = [item['type']['value'] for item in query_types]
    type_names = [item.split("/")[-1] for item in types]
    with open("database/freebase-info/freebase_type_en.txt", "w") as f1:
        f1.write("\n".join(type_names))

def read_txt(filename):

    with open(filename, 'r', encoding='utf-8') as file:
        # readlines() 返回文件中所有行的一个列表
        lines_list = file.readlines()

    clean_lines_list = [line.strip() for line in lines_list]
    print(clean_lines_list[0])
    return clean_lines_list

def cache_type_vectors_batch():
    """
    database/freebase-info/freebase_type_en.txt
    """
    data = read_txt("database/freebase-info/freebase_type_en.txt")
    out_f = f"database/cache_vector_db_fb/type-openai-split-to-words.pkl"

    preds = data
    
    # preds = [p.replace(".", " ").replace("_", " ") for p in preds]
    logger.info(f"Total {len(preds)} predicates")

    id_vecs = {}
    for batch_preds in tqdm(
        [preds[i : i + 1000] for i in range(0, len(preds), 1000)],
        desc="cache type vectors",
    ):
        _batch_preds = [p.replace(".", " ").replace("_", " ") for p in batch_preds]
        batch_vec = get_embedding_batch(_batch_preds)
        assert len(batch_vec) == len(batch_preds)
        for _p, vec in zip(batch_preds, batch_vec):
            id_vecs[_p] = vec
        # key: val --> education.education, embedding(eudcation education)

    print(len(id_vecs.keys()))
    save_to_pkl(id_vecs, out_f)


# ----------------------------------------------------------------------- #


def get_chroma_client():
    import chromadb

    vec_client = chromadb.PersistentClient(path="./database/db_vector_chroma_fb_v2")
    return vec_client


vec_client = get_chroma_client()

CHROMA_TYPE_NAME = "type-openai-split-to-words"


def vectorization_chroma_type(name, vec_f):
    id_vecs = pickle.load(open(vec_f, "rb"))

    embeddings, ids, documents = [], [], []
    for idx, (p, vec) in enumerate(id_vecs.items()):
        embeddings.append(vec)
        ids.append(str(idx))
        documents.append(p)

    names = [i.name for i in vec_client.list_collections()]
    if name in names:
        vec_client.delete_collection(name)
    collection = vec_client.create_collection(name)
    for i in range(0, len(ids), 5000):
        collection.add(
            embeddings=embeddings[i : i + 5000],
            ids=ids[i : i + 5000],
            documents=documents[i : i + 5000],
        )
    count = collection.count()
    logger.info(f"{name}: {count}")


if __name__ == "__main__":
    get_type()
    print('find all type in freebase done.')
    cache_type_vectors_batch()
    print('cache types into vector done.')
    vectorization_chroma_type(
        name=CHROMA_TYPE_NAME,
        vec_f="database/cache_vector_db_fb/type-openai-split-to-words.pkl",
    )
    print('vectorization_chroma_type done.')
    # for test.
    vec_client = get_chroma_client()
    collection_predicate = vec_client.get_collection(name=CHROMA_TYPE_NAME)
    one_hop_emb = get_embedding('education')
    _predicates_vec = collection_predicate.query(
                query_embeddings=one_hop_emb,
                n_results=5,
            )
    print(_predicates_vec['documents'][0])

