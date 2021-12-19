import os
from elasticsearch import Elasticsearch, helpers, NotFoundError
import csv
import time
import tqdm
import requests
import os
import time
import requests
import tarfile
import subprocess
from typing import List, Dict
import logging
logger = logging.getLogger(__name__)
logging.getLogger('elasticsearch').setLevel(logging.CRITICAL)  # muting logging from ES
logging.basicConfig(
    format='%(asctime)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO
)


class ElasticSearchBM25(object):
    """
    Connect to the Elasticsearch service when both valid `host` and `port_http` indicated or create a new one via docker when `host` is None.
    :param corpus: A mapping from IDs to docs.
    :param index_name: Name of the elasticsearch index.
    :param reindexing: Whether to re-index the documents if the index exists.
    :param port_http: The HTTP port of the elasticsearch service.
    :param port_tcp: The TCP port of the elasticsearch service.
    :param host: The host address of the elasticsearch service. If set None, an ES docker container will be started with the indicated port numbers, `port_http` and `port_tcp` exposed.
    :param service_type: When starting ES service needed, use either "docker" to start a new ES docker container or "executable" to download executable ES and run.
    :param es_version: Indicating the elasticsearch version for the docker container.
    :param timeout: Timeout (in seconds) at the ES-service side.
    :param max_waiting: Maximum time (in seconds) to wait for starting the elasticsearch docker container.
    :param cache_dir: Cache directory for downloading the ES executable if needed.
    """
    def __init__(
        self, 
        corpus: Dict[str, str], 
        host: str,
        port_http: str,
        suffix: str,
        index_name: str='one_trial', 
        reindexing: bool=True, 
        es_version: str='7.15.1',
        timeout: int=100,
        max_waiting: int=100,
        cache_dir: str='/tmp'
    ):
        self._wait_and_check(host, port_http, suffix, max_waiting)
        logger.info(f'Successfully reached out to ES service at {host}:{port_http}')
        url = 'http://{host}:{port_http}/{suffix}'
        print(url)
        es = Elasticsearch([url], timeout=timeout)
        logger.info(f'Successfully built connection to ES service at {host}:{port_http}')
        self.es = es
        if es.indices.exists(index=index_name):
            if reindexing:
                logger.info(f'Index {index_name} found and it will be indexed again since reindexing=True')
                es.indices.delete(index=index_name)
        else:
            logger.info(f'No index found and now do indexing')
            self._index_corpus(corpus, index_name)
        self.index_name = index_name
        logger.info('All set up.')

    def _check_service_running(self, host, port, suffix) -> bool:
        """
        Check whether the ES service is reachable.
        :param host: The host address.
        :param port: The HTTP port.
        :return: Whether the ES service is reachable.
        """
        try:
            return requests.get(f'http://{host}:{port}/{suffix}').status_code == 200
        except:
            return False
    
    def _wait_and_check(self, host, port, suffix, max_waiting) -> bool:
        logger.info(f'Waiting for the ES service to be well started. Maximum time waiting: {max_waiting}s')
        timeout = True
        for _ in tqdm.trange(max_waiting):
            if self._check_service_running(host, port, suffix):
                timeout = False
                break
            time.sleep(1)
        assert timeout == False, 'Timeout to start the ES docker container or connect to the ES service, ' + \
            'please increase max_waiting or check the idling ES services ' + \
            '(starting multiple ES instances from ES executable is not allowed)'

    def _index_corpus(self, corpus, index_name):
        """
        Index the corpus.
        :param corpus: A mapping from document ID to documents.
        :param index_name: The name of the target ES index.
        """
        es_index = {
            "mappings": {
                "properties": {
                        "document": {
                            "type": "text"
                        },
                }
            }
        }
        self.es.indices.create(index=index_name, body=es_index, ignore=[400])
        ndocuments = len(corpus)
        dids, documents = list(corpus.keys()), list(corpus.values())
        chunk_size = 500
        pbar = tqdm.trange(0, ndocuments, chunk_size)
        for begin in pbar:
            did_chunk = dids[begin:begin+chunk_size]
            document_chunk = documents[begin:begin+chunk_size]
            bulk_data = [{
                "_index": index_name,
                "_id": did,
                "_source": {
                    "document": documnt,
                }
            } for did, documnt in zip(did_chunk, document_chunk)]
            helpers.bulk(self.es, bulk_data)
        logger.info(f'Indexing work done: {ndocuments} documents indexed') 

    def query(self, query: str, topk, return_scores=False) -> Dict[str, str]:
        """
        Search for a given query.
        :param query: The query text.
        :param topk: Specifying how many top documents to return. Should less than 10000.
        :param return_scores: Whether to return the scores.
        :return: Ranked documents, a mapping from IDs to the documents (and also the scores, a mapping from IDs to scores). 
        """
        assert topk <= 10000, '`topk` is too large!'
        result = self.es.search(index=self.index_name, size=min(topk, 10000), body={
            "query": 
            {
                "match": {
                    "document": query
                }
            }
        })
        hits = result['hits']['hits']
        documents_ranked = {hit['_id']: hit['_source']['document'] for hit in hits}
        if return_scores:
            scores_ranked = {hit['_id']: hit['_score'] for hit in hits}
            return documents_ranked, scores_ranked
        else:
            return documents_ranked
    
    def score(self, query: str, document_ids: List[int], max_ntries=60) -> Dict[str, str]:
        """
        Scoring a query against the given documents (IDs).
        :param query: The query text.
        :param document_ids: The document IDs.
        :param max_ntries: Maximum time (in seconds) for trying.
        :return: The mapping from IDs to scores.
        """
        for i in range(max_ntries):
            try:
                scores = {}
                for document_id in document_ids:
                    result = self.es.explain(index=self.index_name, id=document_id, body={
                        "query": 
                        {
                            "match": {
                                "document": query
                            }
                        }
                    })
                    scores[document_id] = result['explanation']['value']
                return scores
            except NotFoundError as e:
                if i == max_ntries:
                    raise e
                logger.info(f'NotFoundError, now re-trying ({i+1}/{max_ntries}).')
                time.sleep(1)
                
    def delete_index(self):
        """
        Delete the used index.
        """
        if self.es.indices.exists(index=self.index_name):
            logger.info(f'Delete "{self.index_name}": {self.es.indices.delete(self.index_name)}')
        else:
            logger.warning(f'Index "{self.index_name}" does not exist!')