#!/usr/bin/env python3

import json
import requests
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk

def upload_patterns():
    # ES connection
    ES_HOST = "95.217.84.234"
    ES_PORT = 9200
    ES_USER = "elastic"
    ES_PASSWORD = "[REDACTED]"
    ES_INDEX = "ai_service_ac_patterns"

    es = Elasticsearch(
        hosts=[f"http://{ES_HOST}:{ES_PORT}"],
        http_auth=(ES_USER, ES_PASSWORD),
        verify_certs=False,
        timeout=60
    )

    print("🗑️ Deleting existing patterns...")
    try:
        es.indices.delete(index=ES_INDEX)
        print("✅ Deleted existing index")
    except:
        print("⚠️ Index didn't exist")

    print("📋 Creating index with proper mapping...")
    mapping = {
        "mappings": {
            "properties": {
                "pattern": {"type": "keyword"},
                "canonical": {"type": "text"},
                "tier": {"type": "integer"},
                "pattern_type": {"type": "keyword"},
                "language": {"type": "keyword"},
                "confidence": {"type": "float"},
                "source_field": {"type": "keyword"},
                "entity_id": {"type": "keyword"},
                "entity_type": {"type": "keyword"},
                "hints": {"type": "object"}
            }
        },
        "settings": {
            "number_of_shards": 1,
            "number_of_replicas": 0
        }
    }

    es.indices.create(index=ES_INDEX, body=mapping)
    print("✅ Created index with mapping")

    print("📤 Loading patterns from file...")
    with open('patterns_with_partial_names.json', 'r', encoding='utf-8') as f:
        patterns = json.load(f)

    print(f"📊 Uploading {len(patterns)} patterns...")

    def pattern_to_doc(pattern):
        return {
            "_index": ES_INDEX,
            "_source": {
                "pattern": pattern["pattern"],
                "canonical": pattern["canonical"],
                "tier": pattern["metadata"]["tier"],
                "pattern_type": pattern["metadata"]["pattern_type"],
                "language": pattern["metadata"]["language"],
                "confidence": pattern["metadata"]["confidence"],
                "source_field": pattern["metadata"]["source_field"],
                "entity_id": pattern["entity_id"],
                "entity_type": pattern["entity_type"],
                "hints": pattern["metadata"]["hints"]
            }
        }

    docs = [pattern_to_doc(p) for p in patterns]

    # Upload in chunks
    success_count = 0
    for i in range(0, len(docs), 1000):
        chunk = docs[i:i+1000]
        results = bulk(es, chunk, index=ES_INDEX)
        success_count += results[0]
        if i % 10000 == 0:
            print(f"  📤 Uploaded {i + len(chunk)}/{len(docs)} patterns...")

    print(f"✅ Upload complete! {success_count} patterns uploaded")

    # Test search for Poroshenko partial pattern
    print("\n🔍 Testing search for partial patterns...")

    test_queries = [
        "порошенко петро",
        "Порошенко Петро",
        "порошенко",
        "Порошенко"
    ]

    for query in test_queries:
        search_query = {
            "query": {
                "bool": {
                    "should": [
                        {
                            "wildcard": {
                                "pattern": {
                                    "value": f"*{query.lower()}*",
                                    "case_insensitive": True
                                }
                            }
                        },
                        {
                            "term": {
                                "pattern": query.lower()
                            }
                        }
                    ],
                    "minimum_should_match": 1
                }
            }
        }

        result = es.search(index=ES_INDEX, body=search_query, size=5)
        hits = result['hits']['total']['value']
        print(f"  Query '{query}': {hits} matches")

        if hits > 0:
            for hit in result['hits']['hits'][:3]:
                pattern = hit['_source']['pattern']
                pattern_type = hit['_source']['pattern_type']
                print(f"    - '{pattern}' ({pattern_type})")

if __name__ == "__main__":
    upload_patterns()