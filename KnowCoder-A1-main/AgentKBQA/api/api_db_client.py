from traceback import print_exc
import time
import requests

API_SERVER_FB = "http://localhost:9901"
requests.packages.urllib3.disable_warnings()


def get_actions_api(db):
    if db == "kqapro":
        _base = API_SERVER_KQAPRO
        n_results = 10
    elif db == "fb":
        _base = API_SERVER_FB
        n_results = 10
    elif db == "metaqa":
        _base = API_SERVER_METAQA
        n_results = 10
    else:
        raise ValueError(f"db: {db} not supported.")

    base_url = f"{_base}/{db}/"

    def test():
        url = base_url + "test"
        response = requests.get(url, timeout=5, verify=False)
        assert response.status_code == 200, f"Connection to {db} failed."

    test()

    def SearchNodes(query="", n_results=n_results):
        url = base_url + "SearchNodes"
        data = {
            "query": query,
            "n_results": n_results,
        }
        response = requests.post(url, data=data, timeout=150, verify=False)
        return response.json()
    def SearchTypes(query="", n_results=n_results):
        url = base_url + "SearchTypes"
        data = {
            "query": query,
            "n_results": n_results,
        }
        response = requests.post(url, data=data, timeout=150, verify=False)
        return response.json()

    def SearchGraphPatterns(sparql="", semantic="", topN_return=10, return_fact_triple=True):
        """
        def SearchGraphPatterns(sparql: str = None, semantic: str = None, topN_vec=100, return_fact_triple=False):
        """
        url = base_url + "SearchGraphPatterns"
        data = {
            "sparql": sparql,
            "semantic": semantic,
            "return_fact_triple": return_fact_triple,
            "topN_return": topN_return,
        }
        try:
            response = requests.post(url, data=data, timeout=150, verify=False)
            return response.json()
        except Exception as e:
            print_exc()
            print("data,", data)
            return None

    def ExecuteSPARQL(sparql="", str_mode=True):
        url = base_url + "ExecuteSPARQL"
        data = {
            "sparql": sparql,
            "str_mode": str_mode,
        }
        response = requests.post(url, data=data, timeout=150, verify=False)
        return response.json()

    return SearchNodes, SearchTypes, SearchGraphPatterns, ExecuteSPARQL


def test_fb():
    SearchNodes, SearchTypes, SearchGraphPatterns, ExecuteSPARQL = get_actions_api("fb")

    # print(SearchNodes("electronic dance music"))
    s = SearchGraphPatterns(
        'SELECT ?e WHERE { ?e ns:type.object.name "Germany"@en . }',
        semantic="official language",
    )
    print(s)



if __name__ == "__main__":
    # python tools/api_db_client.py
    test_fb()
