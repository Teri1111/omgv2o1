# DB and Tool Setup

This repository is a modified version for a knowledge graph question answering service, based on the code from the ACL 2024 paper **"Interactive-KBQA: Multi-Turn Interactions for Knowledge Base Question Answering with Large Language Models"**.

The original paper and code can be found at:

  * **Paper**: [arXiv:2402.15131](https://arxiv.org/abs/2402.15131)
  * **Original Repository**: [Interactive-KBQA](https://github.com/jimxionggm/interactive-kbqa)

## Overview

This modified version focuses on the **Freebase (FB) database** for knowledge base question answering. We have extended the original code with `SearchTypes` tools to provide support for the GrailQA dataset.

### Prerequisites

  * Python 3.10
  * Virtuoso Open-Source Edition

You can install the required Python packages using:

```bash
pip install gdown chromadb requests
```

-----

### Install Virtuoso

1.  Download the package `virtuoso-opensource.x86_64-generic_glibc25-linux-gnu.tar.gz` from the link below:
    [virtuoso-opensource](https://github.com/openlink/virtuoso-opensource/releases)

2.  After downloading, extract the archive (you can delete the `.tar.gz` file afterward).

    ```bash
    tar -xvzf virtuoso-opensource.x86_64-generic_glibc25-linux-gnu.tar.gz
    ```

3.  Add the environment variables to your `~/.bashrc` file. **Remember to replace `<path-to-your-virtuoso-directory>`** with the absolute path where you extracted the files.

    ```bash
    # Add these lines to ~/.bashrc
    export VIRTUOSO_HOME=<path-to-your-virtuoso-directory>/virtuoso-opensource
    export PATH=.:${VIRTUOSO_HOME}/bin:$PATH
    ```

4.  Apply the changes to your current session or restart the terminal.

    ```bash
    source ~/.bashrc
    ```

Virtuoso installation is complete.

-----

## Freebase Database Setup

For detailed database setup instructions, please refer to [Original Database and Tool Setup](https://www.google.com/search?q=./doc/db_and_tool_setup.md).

> **Note:** All commands below assume you are running them from the root directory of this project.

### 1\. Database Installation and Indexing

#### Step 1: Download Knowledge Base

First, download the Freebase knowledge base file using `gdown`.

```bash
gdown "https://drive.google.com/uc?id=1-chTQ-8UzNQOrnsvONAJvPYk5TZ674X0"
```

If you cannot download via gdown, please download it from the webpage and then upload it.

#### Step 2: Place Database File

Next, create the target directory and move the downloaded `fb_filter_eng_fix_literal.gz` file into it.

```bash
# Create the target directory
mkdir -p database

# Move the file
mv <path-to-downloaded-file>/fb_filter_eng_fix_literal.gz database/
```

#### Step 3: Configure and Start Virtuoso for Indexing

Before starting, you must modify the `tool_prepare/virtuoso_for_indexing.ini` file.

Set the `DirsAllowed` variable to the **absolute path** of your project's root directory. This is crucial for `isql` to find the database file in the next step. You can get the path by running `pwd` in your project's root.

**Example `virtuoso_for_indexing.ini` modification:**

```ini
...
[Parameters]
...
DirsAllowed = /home/user/your_project_name
...
```

(Optional: For optimal performance, you may need to adjust `numberOfBuffers` and `maxDirtyBuffers` based on your server's specifications.)

Start the Virtuoso server in a terminal session:

```bash
virtuoso-t -df -c tool_prepare/virtuoso_for_indexing.ini
```

#### Step 4: Build Knowledge Base via ISQL

In a **separate terminal**, execute the following command to build the knowledge base:

```bash
isql 1111 dba dba
```

This will open the SQL interactive mode. Paste the following commands directly into the `isql` prompt.

```sql
DB.DBA.TTLP_MT(gz_file_open ('database/fb_filter_eng_fix_literal.gz'), '', 'http://freebase.com', 128);
checkpoint;
exit;
```

The indexing process is time-consuming (10-20 hours).

#### Step 5: Verify Indexing (Optional)

After indexing completes, you can verify the count of triples. Re-run `isql 1111 dba dba` and execute:

```sql
SPARQL select count(*) from <http://freebase.com> where {?s ?r ?o};
```

A successful import will return a large number (e.g., `954533166`).

-----

### 2\. Running the Virtuoso Service

After the initial indexing is complete, you can start the Virtuoso database server for regular use.

```bash
virtuoso-t -df -c tool_prepare/virtuoso_for_indexing.ini
```

The Freebase database will be accessible at `http://localhost:9501/sparql` (or your configured port).

-----

## Tool Preparation

### 1\. Preprocess Entity Names and Predicates

```bash
# Cache entity names
python tool_prepare/fb_cache_entity_en.py

# Cache predicates
python tool_prepare/fb_cache_predicate.py
```

This will generate:

  * `database/freebase-info/freebase_entity_name_en.txt`
  * `database/freebase-info/predicate_freq.json`
  * `database/freebase-info/cvt_predicate_onehop.jsonl`

### 2\. Cache Vector Representations

```bash
python tool_prepare/fb_vectorization.py
python tool_prepare/fb_vectorize_type.py
```

### 3\. NEW: FACC1 Entity Popularity for Faster Tool Execution

#### Step 1: Download FACC1

Download the [FACC1](https://github.com/dki-lab/GrailQA/tree/main/entity_linker/data) data to the following directory:

```
database/freebase-info/surface_map_file_freebase_complete_all_mention
```

#### Step 2: Cache FACC1 Entity Popularity

Cache FACC1 entity popularity using sqlite3:

```bash
python tool_prepare/facc1.py
```

If everything goes smoothly, you should see the following output:

```
['m.0fs04cs', 'm.0zjywz4', 'm.0mtgjq8', 'm.0dss9vb', 'm.0msygvy']
```

-----

## Start Knowledge Base Tool Service

### 0\. Configure Embedding API

The tool service uses an OpenAI-compatible embedding API for vector search. You need to configure the API key and endpoint in `tool/openai.py` (lines 23-28):

```python
DEFAULT_KEY = "your-api-key-here"
client = OpenAI(api_key=DEFAULT_KEY, base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")
```

  * `DEFAULT_KEY`: Your API key. The default uses Alibaba Cloud DashScope. You can also set it via the environment variable `OPENAI_API_KEY`.
  * `base_url`: The API endpoint. Change this if you use a different provider.
  * The embedding model used is `text-embedding-v2` (defined in the `get_embedding` function at line 214).

> **If you cannot access external networks**, you can deploy a local embedding model and serve it as an OpenAI-compatible API:
>
> ```bash
> vllm serve BAAI/bge-base-en-v1.5 --task embed --port 8080
> ```
>
> Then update `tool/openai.py`:
>
> ```python
> DEFAULT_KEY = "not-needed"
> client = OpenAI(api_key=DEFAULT_KEY, base_url="http://127.0.0.1:8080/v1")
> ```
>
> **Note:** Make sure to also update the `model=` parameter in the `get_embedding` and `get_embedding_batch` functions (lines 214, 249, 257) to match your deployed model name. The embedding dimension must be consistent with what was used during the vector cache preparation step (Section "Tool Preparation > 2. Cache Vector Representations").

-----

### 1\. Test the Tools

First, verify that all tools are configured correctly and can connect to the database:

```bash
python tool/actions_fb.py
```

Expected output:

```
Connected to Freebase successfully.
Loaded 0 timeout queries.
['Saint Lucy']
```

### 2\. Start HTTP API Server

Start the Freebase tool APIs as a background service.

```bash
# Using screen (recommended)
screen -S api-fb -d -m
screen -S api-fb -X stuff "python api/api_db_server.py --db fb --port 9901
"

# Or run directly in the foreground
python api/api_db_server.py --db fb --port 9901
```

The Freebase tool APIs will be available at `http://localhost:9901`.

-----

## Test the API

You can test the running service using `curl`:

```bash
# Test SearchGraphPatterns
curl -X POST "http://localhost:9901/fb/SearchGraphPatterns" \
    -H "Content-Type: application/x-www-form-urlencoded" \
    -d "sparql=SELECT ?e WHERE { ?e ns:type.object.name \"Jerry Jones\"@en }&semantic=owned by&topN_return=10"

# Test ExecuteSPARQL
curl -X POST "http://localhost:9901/fb/ExecuteSPARQL" \
    -H "Content-Type: application/x-www-form-urlencoded" \
    -d "sparql=SELECT DISTINCT ?x WHERE { ?e0 ns:type.object.name \"Tom Hanks\"@en . ?e0 ns:award.award_winner.awards_won ?cvt0 . ?cvt0 ns:award.award_honor.award ?x . }&str_mode=false"

# Test SearchTypes
curl -X POST "http://localhost:9901/fb/SearchTypes" \
    -H "Content-Type: application/x-www-form-urlencoded" \
    -d "query=education" \
    -w "\nTotal time: %{time_total}s\n"
```

## Acknowledgments

This code is based on the excellent work by the authors of "Interactive-KBQA: Multi-Turn Interactions for Knowledge Base Question Answering with Large Language Models". We are grateful for their contribution to the KBQA community.