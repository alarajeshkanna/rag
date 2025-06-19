Process a single document:
python main1.py --source https://example.com --loader URL --chunking SEMANTIC --embedding BGE_LARGE

Process a batch of documents:
python main.py --batch documents.json --chunking MARKDOWN

Interactive mode:
python main1.py --interactive

With custom config:
python main1.py --config custom_config.json --source document.pdf





python main2.py \
  --source "https://example.com/article" \
  --loader url \
  --chunking recursive \
  --embedding bge-large-en-v1.5 \
  --chunk_size 800 \
  --chunk_overlap 100 \
  --extract_keywords


python main2.py \
  --source "./data/sample.pdf" \
  --loader pdf \
  --chunking semantic \
  --embedding sentence-transformers/all-MiniLM-L6-v2
