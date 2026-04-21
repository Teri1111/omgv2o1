import os

CWQ_DATA_DIR = "/data/gt/omg/data/CWQ/original/"
OMGV1_DIR = "/data/gt/omgv1"
OMG_DIR = "/data/gt/omg"

SPARQL_ENDPOINT = "http://192.168.1.153:8890/sparql"

T5_CKPT = "/data/gt/omg/models/t5_ckpt/cwq/Generator-epoch68.ckpt"

PYTHON_LF = "/data/gt/envs/lf_gjq/bin/python"
PYTHON_T5 = "/data/gt/envs/cwq_brr/bin/python"

CWQ_CONTEXT = "/data/gt/omg/data/CWQ/search_mid/CWQ_context_test.json"
NAMEDICT = "/data/gt/omg/data/CWQ/original/namedict.json"

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
