python main.py \
  --source "https://example.com/article" \
  --loader url \
  --chunking recursive \
  --embedding bge-large-en-v1.5 \
  --chunk_size 800 \
  --chunk_overlap 100 \
  --extract_keywords


python main.py \
  --source "./data/sample.pdf" \
  --loader pdf \
  --chunking semantic \
  --embedding sentence-transformers/all-MiniLM-L6-v2
