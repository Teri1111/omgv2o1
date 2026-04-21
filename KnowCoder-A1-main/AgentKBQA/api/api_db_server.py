import argparse
import asyncio
import time
import os, sys
import logging
from datetime import datetime
from typing import Annotated

import uvicorn
from fastapi import FastAPI, Form

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_dir)
from common.constant import API_SERVER_FB, API_SERVER_KQAPRO, API_SERVER_METAQA
from tool.openai import get_embedding

"""
kqapro
python tools/api_db_server.py --db kqapro

fb
python tools/api_db_server.py --db fb

metaqa
python tools/api_db_server.py --db metaqa
"""

argparser = argparse.ArgumentParser()
argparser.add_argument("--db", type=str, required=True, help="kqapro, fb, metaqa")
argparser.add_argument("--port", type=str, required=True, help="port")
args = argparser.parse_args()

if args.db == "kqapro":
    _url = API_SERVER_KQAPRO
    from tool.actions_kqapro import init_kqapro_actions as init_actions
elif args.db == "fb":
    _url = API_SERVER_FB
    from tool.actions_fb import init_fb_actions as init_actions
elif args.db == "metaqa":
    _url = API_SERVER_METAQA
    from tool.actions_metaqa import init_metaqa_actions as init_actions
else:
    raise ValueError(f"db: {args.db} not supported.")

args.host, _ = _url.split("http://")[-1].split(":")
args.port = int(args.port)
print(f"db: {args.db}, host: {args.host}, port: {args.port}")

SearchNodes, SearchTypes, SearchGraphPatterns, ExecuteSPARQL = init_actions()

app = FastAPI()

api_logger = logging.getLogger("api_timing")

SLOW_THRESHOLD = 10


@app.post(f"/{args.db}/SearchNodes")
async def _SearchNodes(query: Annotated[str, Form()] = "", n_results: Annotated[int, Form()] = 10):
    start = time.time()
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, SearchNodes, query, n_results)
    elapsed = time.time() - start
    tag = "SLOW" if elapsed > SLOW_THRESHOLD else "OK"
    api_logger.info(f"[{tag}] SearchNodes | {elapsed:.2f}s | query={query[:100]}")
    return result


@app.post(f"/{args.db}/SearchTypes")
async def _SearchTypes(query: Annotated[str, Form()] = "", n_results: Annotated[int, Form()] = 10):
    start = time.time()
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, SearchTypes, query, n_results)
    elapsed = time.time() - start
    tag = "SLOW" if elapsed > SLOW_THRESHOLD else "OK"
    api_logger.info(f"[{tag}] SearchTypes | {elapsed:.2f}s | query={query[:100]}")
    return result


@app.post(f"/{args.db}/SearchGraphPatterns")
async def _SearchGraphPatterns(
    sparql: Annotated[str, Form()] = "",
    semantic: Annotated[str, Form()] = "",
    topN_vec: Annotated[int, Form()] = 400,
    topN_return: Annotated[int, Form()] = 10,
):
    start = time.time()
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, SearchGraphPatterns, sparql, semantic, topN_vec, topN_return)
    elapsed = time.time() - start
    tag = "SLOW" if elapsed > SLOW_THRESHOLD else "OK"
    api_logger.info(f"[{tag}] SearchGraphPatterns | {elapsed:.2f}s | sparql={sparql[:150]} | semantic={semantic[:80]}")
    return result


@app.post(f"/{args.db}/ExecuteSPARQL")
async def _ExecuteSPARQL(sparql: Annotated[str, Form()] = "", str_mode: Annotated[bool, Form()] = True):
    start = time.time()
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, ExecuteSPARQL, sparql, str_mode)
    elapsed = time.time() - start
    tag = "SLOW" if elapsed > SLOW_THRESHOLD else "OK"
    api_logger.info(f"[{tag}] ExecuteSPARQL | {elapsed:.2f}s | sparql={sparql[:200]}")
    return result


@app.get(f"/{args.db}/test")
async def _test():
    current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    return {"status": "ok", "start_time": current_time}


@app.post("/get_embedding")
async def _get_embedding(
    text: Annotated[str, Form()],
    model: Annotated[str, Form()] = "miniLM",
):
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, get_embedding, text, model)
    return result


if __name__ == "__main__":
    log_dir = os.path.join(base_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"api_db_server_{args.db}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

    LOG_CONFIG = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "standard": {
                "format": "%(asctime)s [%(name)s] %(levelname)s: %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S",
            },
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "formatter": "standard",
                "stream": "ext://sys.stdout",
            },
            "file": {
                "class": "logging.FileHandler",
                "formatter": "standard",
                "filename": log_file,
                "encoding": "utf-8",
            },
        },
        "loggers": {
            "uvicorn": {"handlers": ["console", "file"], "level": "INFO", "propagate": False},
            "uvicorn.error": {"handlers": ["console", "file"], "level": "INFO", "propagate": False},
            "uvicorn.access": {"handlers": ["console", "file"], "level": "INFO", "propagate": False},
            "api_timing": {"handlers": ["console", "file"], "level": "INFO", "propagate": False},
        },
    }

    print(f"[LOG] Log file: {log_file}")

    uvicorn.run(
        "api_db_server:app",
        host=args.host,
        port=args.port,
        workers=8,
        timeout_keep_alive=150,
        log_config=LOG_CONFIG,
    )
