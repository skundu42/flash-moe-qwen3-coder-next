#!/usr/bin/env python3
"""Export Hugging Face tokenizer assets for the C runtime.

Binary format:
  Header:
    magic: "BPET" (4 bytes)
    version: uint32
    vocab_size: uint32
    num_merges: uint32
    num_added: uint32
  Vocab section (sorted by token_id):
    For each entry: uint32 token_id, uint16 str_len, char[str_len] (UTF-8 bytes of the BPE string)
  Merges section (ordered by priority, index 0 = highest priority):
    For each entry: uint16 len_a, char[len_a], uint16 len_b, char[len_b]
  Added tokens section:
    For each entry: uint32 token_id, uint16 str_len, char[str_len]
"""
import argparse
import json
import os
import struct


def parse_args():
    parser = argparse.ArgumentParser(description="Export tokenizer.json into tokenizer.bin and vocab.bin")
    default_model_dir = os.environ.get("QWEN3_CODER_NEXT_MODEL_PATH", "./Qwen3-Coder-Next")
    parser.add_argument(
        "--model-dir",
        default=default_model_dir,
        help="Model directory containing tokenizer.json",
    )
    parser.add_argument(
        "--tokenizer-json",
        default=None,
        help="Explicit tokenizer.json path (overrides --model-dir/tokenizer.json)",
    )
    parser.add_argument(
        "--tokenizer-out",
        default=None,
        help="Output path for tokenizer.bin (default: <model-dir>/tokenizer.bin)",
    )
    parser.add_argument(
        "--vocab-out",
        default=None,
        help="Output path for vocab.bin (default: <model-dir>/vocab.bin)",
    )
    return parser.parse_args()


def normalize_merges(raw_merges):
    merges = []
    for idx, pair in enumerate(raw_merges):
        if isinstance(pair, str):
            parts = pair.split(" ", 1)
            if len(parts) != 2:
                raise ValueError(f"Invalid merge entry at index {idx}: {pair!r}")
            a, b = parts
        elif isinstance(pair, (list, tuple)) and len(pair) == 2:
            a, b = pair
        else:
            raise ValueError(f"Unsupported merge entry at index {idx}: {pair!r}")
        merges.append((a, b))
    return merges


def write_tokenizer_bin(out_path, sorted_vocab, merges, added):
    with open(out_path, 'wb') as f:
        f.write(b'BPET')
        f.write(struct.pack('<I', 1))
        f.write(struct.pack('<I', len(sorted_vocab)))
        f.write(struct.pack('<I', len(merges)))
        f.write(struct.pack('<I', len(added)))

        for token_str, token_id in sorted_vocab:
            token_bytes = token_str.encode('utf-8')
            f.write(struct.pack('<I', token_id))
            f.write(struct.pack('<H', len(token_bytes)))
            f.write(token_bytes)

        for a, b in merges:
            a_bytes = a.encode('utf-8')
            b_bytes = b.encode('utf-8')
            f.write(struct.pack('<H', len(a_bytes)))
            f.write(a_bytes)
            f.write(struct.pack('<H', len(b_bytes)))
            f.write(b_bytes)

        for tok in added:
            token_bytes = tok['content'].encode('utf-8')
            f.write(struct.pack('<I', tok['id']))
            f.write(struct.pack('<H', len(token_bytes)))
            f.write(token_bytes)


def build_byte_unicode_table():
    byte_char = {}
    n = 0
    for b in range(256):
        if (0x21 <= b <= 0x7E) or (0xA1 <= b <= 0xAC) or (0xAE <= b <= 0xFF):
            byte_char[b] = chr(b)
        else:
            byte_char[b] = chr(256 + n)
            n += 1
    return byte_char


BYTE_UNICODE = build_byte_unicode_table()
UNICODE_BYTE = {v: k for k, v in BYTE_UNICODE.items()}


def decode_base_vocab_token(token_str):
    raw = bytearray()
    for ch in token_str:
        b = UNICODE_BYTE.get(ch)
        if b is None:
            # Already a normal codepoint not produced by the GPT-2 byte map.
            raw.extend(ch.encode("utf-8"))
        else:
            raw.append(b)
    return raw.decode("utf-8", errors="replace")


def build_full_decode_table(sorted_vocab, added):
    max_base_id = sorted_vocab[-1][1] if sorted_vocab else -1
    max_added_id = max((tok["id"] for tok in added), default=-1)
    max_id = max(max_base_id, max_added_id)
    contiguous = [None] * (max_id + 1)

    for token_str, token_id in sorted_vocab:
        if token_id < 0:
            raise ValueError(f"Negative token id: {token_id}")
        decoded = decode_base_vocab_token(token_str)
        if contiguous[token_id] is not None and contiguous[token_id] != decoded:
            raise ValueError(f"Conflicting base vocab entry for id {token_id}")
        contiguous[token_id] = decoded

    for tok in added:
        token_id = tok["id"]
        token_str = tok["content"]
        if token_id < 0:
            raise ValueError(f"Negative added token id: {token_id}")
        if contiguous[token_id] is not None and contiguous[token_id] != token_str:
            raise ValueError(
                f"Conflicting added token entry for id {token_id}: "
                f"{contiguous[token_id]!r} vs {token_str!r}"
            )
        contiguous[token_id] = token_str

    missing = [i for i, token_str in enumerate(contiguous) if token_str is None]
    if missing:
        preview = ", ".join(str(x) for x in missing[:8])
        raise ValueError(f"Tokenizer vocab has gaps; cannot write vocab.bin. Missing ids: {preview}")
    return contiguous


def write_vocab_bin(out_path, sorted_vocab, added):
    contiguous = build_full_decode_table(sorted_vocab, added)
    max_id = len(contiguous) - 1

    with open(out_path, 'wb') as f:
        f.write(struct.pack('<I', len(contiguous)))
        f.write(struct.pack('<I', max_id))
        for token_str in contiguous:
            token_bytes = token_str.encode('utf-8')
            f.write(struct.pack('<H', len(token_bytes)))
            f.write(token_bytes)


def main():
    args = parse_args()
    model_dir = args.model_dir
    tok_path = args.tokenizer_json or os.path.join(model_dir, "tokenizer.json")
    tokenizer_out = args.tokenizer_out or os.path.join(model_dir, "tokenizer.bin")
    vocab_out = args.vocab_out or os.path.join(model_dir, "vocab.bin")

    with open(tok_path, 'r', encoding='utf-8') as f:
        t = json.load(f)

    model = t['model']
    vocab = model['vocab']       # str -> int
    merges = normalize_merges(model['merges'])
    added = t['added_tokens']    # list of {id, content, special, ...}

    # Sort vocab by token_id
    sorted_vocab = sorted(vocab.items(), key=lambda x: x[1])

    full_decode_table = build_full_decode_table(sorted_vocab, added)

    write_tokenizer_bin(tokenizer_out, sorted_vocab, merges, added)
    write_vocab_bin(vocab_out, sorted_vocab, added)

    print("Export complete:")
    print(f"  tokenizer.json: {tok_path}")
    print(f"  tokenizer.bin:  {tokenizer_out} ({os.path.getsize(tokenizer_out) / 1024 / 1024:.1f} MB)")
    print(f"  vocab.bin:      {vocab_out} ({os.path.getsize(vocab_out) / 1024 / 1024:.1f} MB)")
    print(f"  Base vocab:     {len(sorted_vocab)}")
    print(f"  Decode entries: {len(full_decode_table)}")
    print(f"  Merge rules:    {len(merges)}")
    print(f"  Added tokens:   {len(added)}")

if __name__ == '__main__':
    main()
