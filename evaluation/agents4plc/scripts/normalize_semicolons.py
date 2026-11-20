"""Insert only missing compound-statement semicolons; compare complete RuSTy ASTs."""
from __future__ import annotations

ENDINGS = {'END_IF', 'END_CASE', 'END_FOR', 'END_WHILE'}


def tokens(text):
    """Yield ST lexical tokens, preserving source offsets and ignoring comments."""
    i, size = 0, len(text)
    while i < size:
        if text[i].isspace():
            i += 1
        elif text.startswith('//', i):
            end = text.find('\n', i + 2)
            i = size if end < 0 else end + 1
        elif text.startswith('(*', i):
            start, depth = i, 1
            i += 2
            while i < size and depth:
                if text.startswith('(*', i): depth += 1; i += 2
                elif text.startswith('*)', i): depth -= 1; i += 2
                else: i += 1
            if depth: raise ValueError(f'unclosed ST comment at {start}')
        elif text[i] in "'\"":
            start, quote = i, text[i]
            i += 1
            while i < size:
                if text[i] == '$': i += 2
                elif text[i] == quote:
                    if i + 1 < size and text[i + 1] == quote: i += 2
                    else: i += 1; break
                else: i += 1
            else: raise ValueError(f'unclosed ST string at {start}')
            yield text[start:i], start, i
        elif text[i] == '{':
            # Compiler pragmas are not executable statement tokens.
            start, depth = i, 1
            i += 1
            while i < size and depth:
                if text[i] == '{': depth += 1
                elif text[i] == '}': depth -= 1
                i += 1
            if depth: raise ValueError(f'unclosed ST pragma at {start}')
        elif text[i].isalpha() or text[i] == '_':
            start = i
            i += 1
            while i < size and (text[i].isalnum() or text[i] == '_'): i += 1
            yield text[start:i], start, i
        else:
            yield text[i], i, i + 1
            i += 1


def normalize(text):
    lexical = list(tokens(text))
    edits = []
    for index, (token, start, end) in enumerate(lexical):
        if token.upper() in ENDINGS and (index + 1 == len(lexical) or lexical[index + 1][0] != ';'):
            edits.append({'character_offset': end, 'byte_offset': len(text[:end].encode()),
                          'keyword': token, 'inserted': ';'})
    candidate = text
    for edit in reversed(edits):
        offset = edit['character_offset']
        candidate = candidate[:offset] + ';' + candidate[offset:]
    return candidate, edits


def verify_insertions(original, candidate, edits):
    expected, expected_edits = normalize(original)
    if candidate != expected or edits != expected_edits:
        raise ValueError('candidate contains changes beyond the exact permitted semicolon insertions')
    second, further = normalize(candidate)
    if second != candidate or further:
        raise ValueError('normalization is not idempotent')


def canonical_ast(text):
    """Drop only typed SourceLocation metadata; preserve every other AST byte.

    AST pairs use the very same isolated input path, so file names need no removal.
    Quoted user strings cannot be mistaken for SourceLocation debug structures.
    """
    i, size, output, removed = 0, len(text), [], 0
    while i < size:
        if text[i] == '"':
            start = i; i += 1
            while i < size:
                if text[i] == '\\': i += 2
                elif text[i] == '"': i += 1; break
                else: i += 1
            else: raise ValueError('unterminated Rust debug string')
            output.append(text[start:i])
        elif text.startswith('SourceLocation {', i) and (i == 0 or not (text[i - 1].isalnum() or text[i - 1] == '_')):
            i += len('SourceLocation {'); depth = 1
            while i < size and depth:
                if text[i] == '"':
                    i += 1
                    while i < size:
                        if text[i] == '\\': i += 2
                        elif text[i] == '"': i += 1; break
                        else: i += 1
                    else: raise ValueError('unterminated location string')
                else:
                    if text[i] == '{': depth += 1
                    elif text[i] == '}': depth -= 1
                    i += 1
            if depth: raise ValueError('unterminated SourceLocation metadata')
            output.append('SourceLocation { <source metadata> }'); removed += 1
        else:
            output.append(text[i]); i += 1
    return ''.join(output), removed
