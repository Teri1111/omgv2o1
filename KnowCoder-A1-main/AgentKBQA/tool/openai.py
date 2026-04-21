import json
import os
import sqlite3
import threading
from typing import List, Optional

import httpx
import os,sys
_cpath_ = sys.path[0] 
sys.path.remove(_cpath_) 
from openai import OpenAI
sys.path.insert(0, _cpath_) 
from tenacity import retry, stop_after_attempt, wait_fixed

magic_url, http_client = None, None

# :-)
# magic_url = "http://127.0.0.1:7893"

if magic_url:
    http_client = httpx.Client(proxies=magic_url)

DEFAULT_KEY = os.getenv("OPENAI_API_KEY")
assert DEFAULT_KEY is not None, "OPENAI_API_KEY is None, use `export OPENAI_API_KEY=your_key` to set it."


client = OpenAI(api_key=DEFAULT_KEY, base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")


@retry(wait=wait_fixed(2), stop=stop_after_attempt(2))
def chatgpt(
    prompt="Hello!",
    system_content="You are an AI assistant.",
    messages=None,
    model="deepseek-chat",
    temperature=0,
    top_p=1,
    n=1,
    stop=None,  # ["\n"],
    max_tokens=256,
    presence_penalty=0,
    frequency_penalty=0,
    logit_bias={},
):
    """
    role:
        The role of the author of this message. One of `system`, `user`, or `assistant`.
    temperature:
        What sampling temperature to use, between 0 and 2. Higher values like 0.8 will make the output more random, while lower values like 0.2 will make it more focused and deterministic.
        We generally recommend altering this or `top_p` but not both.
    top_p:
        An alternative to sampling with temperature, called nucleus sampling, where the model considers the results of the tokens with top_p probability mass. So 0.1 means only the tokens comprising the top 10% probability mass are considered.
        We generally recommend altering this or `temperature` but not both.

    messages as history usage:
        history = [{"role": "system", "content": "You are an AI assistant."}]

        inp = "Hello!"
        history.append({"role": "user", "content": inp})
        response = chatgpt(messages=history)
        out = response["choices"][0]["message"]["content"]
        history.append({"role": "assistant", "content": out})
        print(json.dumps(history,ensure_ascii=False,indent=4))
    """
    assert model is not None, "model name is None"

    messages = (
        messages
        if messages
        else [
            {"role": "system", "content": system_content},
            {"role": "user", "content": prompt},
        ]
    )

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        top_p=top_p,
        n=1,
        stop=stop,
        max_tokens=max_tokens,
        presence_penalty=presence_penalty,
        frequency_penalty=frequency_penalty,
        logit_bias=logit_bias,
    )
    # content = response["choices"][0]["message"]["content"]
    response = json.loads(response.model_dump_json())
    return response


"""
This is for handling the multi-threading issue of sqlite3.
"""
thread_local = threading.local()

def get_db_path():
    pid = os.getpid()
    # Use environment variable or default to relative path
    base_dir = os.getenv(
        "VECTOR_CACHE_DIR",
        os.path.join(os.path.dirname(os.path.dirname(__file__)), "database", "cache_vector_query")
    )
    os.makedirs(base_dir, exist_ok=True)
    return os.path.join(base_dir, f"local_cache_{pid}.db")

def init_db():
    db_path = get_db_path()
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    with connect_db() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """CREATE TABLE IF NOT EXISTS vec_cache (
                name TEXT PRIMARY KEY,
                vec TEXT NOT NULL
            );"""
        )
        conn.commit()

def connect_db():
    db_path = get_db_path()
    try:
        conn = sqlite3.connect(db_path, timeout=20)
        conn.execute("PRAGMA journal_mode=WAL;") 
        conn.execute("PRAGMA synchronous=NORMAL;") 
        conn.execute("PRAGMA cache_size=10000;") 
        conn.execute("PRAGMA temp_store=MEMORY;") 
        conn.row_factory = sqlite3.Row
        conn.execute("SELECT 1")
        return conn
    except sqlite3.DatabaseError as e:
        print(f"[connect_db] Database error: {e}")
        try:
            if os.path.exists(db_path):
                os.remove(db_path)
        except Exception as remove_error:
            print(f"[connect_db] Failed to remove corrupted database: {remove_error}")
        init_db()
        conn = sqlite3.connect(db_path, timeout=20)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        conn.execute("PRAGMA cache_size=10000;")
        conn.execute("PRAGMA temp_store=MEMORY;")
        conn.row_factory = sqlite3.Row
        return conn
    except Exception as e:
        print(f"[connect_db] Unexpected error: {e}")
        init_db()
        conn = sqlite3.connect(db_path, timeout=20)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        conn.execute("PRAGMA cache_size=10000;")
        conn.execute("PRAGMA temp_store=MEMORY;")
        conn.row_factory = sqlite3.Row
        return conn

def get_vec_cache(name: str) -> Optional[List[float]]:
    try:
        with connect_db() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT vec FROM vec_cache WHERE name=?", (name,))
            res = cursor.fetchone()
            if res:
                return json.loads(res[0])
    except sqlite3.OperationalError as e:
        init_db()
        return get_vec_cache(name)
    except Exception as e:
        print(f"[get_vec_cache] Error: {e}")
    return None

def insert_vec_cache(name: str, vec: List[float]):
    # 确保数据库表已创建
    init_db()
    
    if get_vec_cache(name):
        return

    try:
        with connect_db() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO vec_cache (name, vec) VALUES (?, ?)",
                (name, json.dumps(vec)),
            )
            conn.commit()
    except sqlite3.IntegrityError:
        pass
    except sqlite3.DatabaseError as e:
        print(f"[insert_vec_cache] Database error: {e}")
        try:
            with connect_db() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "INSERT OR REPLACE INTO vec_cache (name, vec) VALUES (?, ?)",
                    (name, json.dumps(vec)),
                )
                conn.commit()
        except Exception as retry_error:
            print(f"[insert_vec_cache] Retry failed: {retry_error}")

def get_embedding(
    text: str,
    model="qwen",
) -> list[float]:
    text_unikey = text + model
    res = get_vec_cache(text_unikey)
    if res:
        assert type(res) == list
        return res
    
    res = client.embeddings.create(model="text-embedding-v2",input=[text], encoding_format="float")
    res = res.data[0].embedding
    insert_vec_cache(text_unikey, res)
    return res


def get_embedding_batch(
    texts: List[str],
    model="qwen",
) -> list[float]:
    # 确保数据库表已创建
    init_db()
    
    cache_results = {}
    try:
        with connect_db() as conn:
            cursor = conn.cursor()
            placeholders = ','.join(['?' for _ in texts])
            cache_keys = [(text + model,) for text in texts]
            cursor.execute(
                f"SELECT name, vec FROM vec_cache WHERE name IN ({placeholders})",
                [key[0] for key in cache_keys]
            )
            for row in cursor.fetchall():
                cache_results[row[0]] = json.loads(row[1])
    except Exception as e:
        print(f"批量读取缓存失败: {e}")
        cache_results = {}

    unseen_texts = [text for text in texts if (text + model) not in cache_results]
    
    if unseen_texts:
        vec_batch = []
        if len(unseen_texts) > 25:
            for batch in [unseen_texts[i : i + 25] for i in range(0, len(unseen_texts), 25)]:
                batch_req = client.embeddings.create(
                    model="text-embedding-v2",
                    input=batch,
                    encoding_format="float"
                )
                vec_batch.extend([i.embedding for i in batch_req.data])
        else:
            batch_req = client.embeddings.create(
                model="text-embedding-v2",
                input=unseen_texts,
                encoding_format="float"
            )
            vec_batch.extend([i.embedding for i in batch_req.data])

        try:
            with connect_db() as conn:
                cursor = conn.cursor()
                for text, vec in zip(unseen_texts, vec_batch):
                    cursor.execute(
                        "INSERT OR REPLACE INTO vec_cache (name, vec) VALUES (?, ?)",
                        (text + model, json.dumps(vec)),
                    )
                conn.commit()
        except Exception as e:
            print(f"batch insert cache failed: {e}")

    # 合并缓存结果和新计算的结果
    result = []
    for text in texts:
        key = text + model
        if key in cache_results:
            result.append(cache_results[key])
        else:
            # 从新计算的向量中获取
            idx = unseen_texts.index(text)
            result.append(vec_batch[idx])
    
    return result

@retry(wait=wait_fixed(2), stop=stop_after_attempt(2))
def get_embedding_openai(
    text: str,
    model="text-embedding-ada-002",
) -> list[float]:
    text_unikey = text + model
    res = get_vec_cache(text_unikey)
    if res:
        assert type(res) == list
        return res
    res = client.embeddings.create(input=[text], model=model).data[0].embedding
    insert_vec_cache(text_unikey, res)
    return res


@retry(wait=wait_fixed(2), stop=stop_after_attempt(2))
def get_embedding_batch_openai(
    texts: List[str],
    model="text-embedding-ada-002",
) -> list[float]:
    unseen_texts: List[str] = []
    for text in texts:
        cache = get_vec_cache(text + model)
        if cache is None:
            unseen_texts.append(text)

    if unseen_texts:
        req = client.embeddings.create(input=unseen_texts, model=model)
        vec_batch = [i.embedding for i in req.data]
        assert len(vec_batch) == len(unseen_texts)
        for unseen_text, vec in zip(unseen_texts, vec_batch):
            insert_vec_cache(unseen_text + model, vec)

    res = [get_vec_cache(text + model) for text in texts]
    assert None not in res
    return res

def test_embedding():
    # text = "hello!! how are you? I am fine."
    # res = get_embedding(text)
    # print(res)
    arr = ['influence influence node influenced', 'influence influence node influenced by', 'media common dedication dedicated to', 'base culturalevent event entity involved', 'people profession people with this profession', 'business employment tenure person', 'book book edition author editor', 'base kwebbase kwconnection subject', 'music composition composer', 'government us vice president to president', 'people person parents', 'base inaugurations inauguration president', 'symbols name source namesakes', 'community discussion thread topic', 'government political party tenure politician', 'government government position held office holder', 'people ethnicity people', 'exhibitions exhibition subjects', 'base famouspets pet ownership owner', 'book written work author', 'education education student', 'people person children', 'location location people born here', 'dataworld gardening hint replaced by', 'film film subjects', 'symbols namesake named after', 'fictional universe fictional character based on', 'architecture structure architect', 'people marriage spouse', 'visual art artwork art subject', 'award award honor award winner', 'government us president vice president', 'organization organization membership member', 'religion religion notable figures', 'user tsegaran random taxonomy entry subject', 'government election winner', 'people sibling relationship sibling', 'kp lw philosopher influenced by', 'base ontologies ontology instance mapping freebase topic', 'base kwebbase kwconnection other', 'time event people involved', 'media common quotation author', 'people appointment appointed by', 'government election campaign candidate', 'base wordnet synset equivalent topic', 'user tfmorris default domain document signatories', 'radio radio program subjects', 'influence peer relationship peers', 'book literary series author s', 'base yupgrade user topics', 'organization organization founders', 'dataworld gardening hint last referenced by', 'event speech or presentation speaker s', 'people place of interment interred here', 'government government position held appointed by', 'people place lived person', 'base kwebbase kwsentence kwtopic', 'book written work subjects', 'influence influence node influenced', 'base kwebbase kwtopic connections from', 'influence influence node influenced by', 'base kwebbase kwtopic has sentences', 'government us vice president to president', 'base kwebbase kwtopic connections to', 'people person parents', 'people person profession', 'people person quotations', 'book author openlibrary id', 'visual art art subject artwork on the subject', 'user rlyeh certainty complete completeproperties', 'fictional universe person in fiction representations in fiction', 'event public speaker speeches or presentations', 'user tfmorris default domain signatory documents signed', 'government us vice president vice president number', 'base inaugurations inauguration speaker inauguration', 'people person quotationsbook id', 'architecture architect structure count', 'people person place of birth', 'book author book editions published', 'organization organization founder organizations founded', 'exhibitions exhibition subject exhibitions created about this subject', 'symbols name source namesakes', 'people person places lived', 'book author works written', 'people person nationality', 'film film subject films', 'type object type', 'base schemastaging context name nickname', 'government u s congressperson thomas id', 'user tsegaran random taxonomy subject entry', 'people person children', 'base ontologies ontology instance equivalent instances', 'symbols namesake named after', 'base famouspets pet owner pets owned', 'architecture architect structures designed', 'government us president presidency number', 'people person date of birth', 'government political appointer appointees', 'base schemastaging context name official name', 'people deceased person place of burial', 'government us president vice president', 'organization organization member member of', 'people deceased person place of death', 'people person sibling s', 'user robert default domain daylife hero image id', 'people person religion', 'nytimes topic uri', 'people person spouse s', 'people deceased person date of death', 'government politician government positions held', 'radio radio subject programs with this subject', 'people person education', 'people person ethnicity', 'award award winner awards won', 'government politician party', 'people appointer appointment made', 'influence influence node peers', 'people person gender', 'kp lw philosophy influencer influencee', 'book author series written or contributed to', 'people person height meters', 'people person employment history', 'base kwebbase kwtopic disciplines', 'base kwebbase kwtopic assessment', 'base kwebbase kwtopic kwtype', 'base kwebbase kwtopic category', 'media common dedicatee dedications', 'music composer compositions', 'government politician election campaigns', 'book book subject works']
    res = get_embedding_batch(arr)
    for item in res:
        print(len(item), end='|')

def test_chatgpt():
    text = "who are you?"
    res = chatgpt(prompt=text, model="gpt-3.5-turbo", timeout=5, max_tokens=32)
    print(res)


if __name__ == "__main__":
    init_db()
    test_embedding()
