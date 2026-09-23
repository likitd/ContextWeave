import os
import config
import tiktoken
from document_loader import discover_files, load_file_as_text

def main():
    folder = config.CONTEXT_FOLDER
    enc = tiktoken.encoding_for_model("gpt-4o")
    file_paths = discover_files(folder)
    file_tokens = []
    for fp in file_paths:
        content = load_file_as_text(fp)
        if content is None:
            continue
        rel = os.path.relpath(fp, folder)
        tokens = len(enc.encode(content))
        file_tokens.append((rel, tokens))
    file_tokens.sort(key=lambda x: x[1], reverse=True)
    total = sum(t for _, t in file_tokens)
    print("File-wise context size (tokens):")
    print("-" * 60)
    for rel, count in file_tokens:
        print(f"  {count:>8}  {rel}")
    print("-" * 60)
    print(f"  {total:>8}  (total, {len(file_tokens)} files)")
    print(f"\nFull context size (all files): {total} tokens")


if __name__ == "__main__":
    main()
