"""Narrow RuSTy ABI corrections, installed on a separately loaded evaluator."""
import ctypes
import re


def install(engine):
    def parse_structs(header, entry, task):
        text = header.read_text(encoding='utf-8')
        bodies = {name: body for body, name in engine.STRUCT_RE.findall(text)}
        match = re.search(engine.FUNCTION_RE_TEMPLATE.format(entry=re.escape(entry)), text)
        if not match:
            raise engine.EntryPointError(f'entry point not exported: {entry}')
        entry_name = match.group(1)
        inouts = {d['name']: d['type'] for d in task['pou_declarations'] if d['section'] == 'inout'}
        built, building = {}, set()

        def build(name):
            resolved = name if name in bodies else name + '_type'
            if resolved in built:
                return built[resolved]
            if resolved not in bodies or resolved in building:
                raise engine.FunctionalError(f'unsupported or recursive C ABI type: {name}')
            building.add(resolved)
            fields = []
            for raw in bodies[resolved].splitlines():
                raw = raw.strip()
                if not raw or raw.startswith('//'):
                    continue
                field = engine.FIELD_RE.fullmatch(raw)
                if not field:
                    raise engine.FunctionalError(f'unsupported C ABI field: {raw}')
                c_name, field_name, dimensions = field.groups()
                c_name = re.sub(r'\s+', '', c_name)
                manifest_inout = resolved == entry_name and field_name in inouts
                # RuSTy emits scalar IN_OUT pointers explicitly, but array IN_OUT
                # headers omit the pointer. Preserve the existing array rule.
                if manifest_inout and not c_name.endswith('*') and not dimensions:
                    raise engine.InterfaceError(f'IN_OUT contract mismatch: {field_name} is a non-pointer scalar in compiled header')
                pointer = c_name.endswith('*') or manifest_inout
                field_type = ctypes.c_void_p if pointer else engine.C_TYPES.get(c_name) or engine.IEC_CTYPES.get(c_name.upper())
                if field_type is None:
                    field_type = build(c_name) if c_name in bodies or c_name + '_type' in bodies else ctypes.c_int32 if c_name.startswith('__') else None
                if field_type is None:
                    raise engine.FunctionalError(f'unsupported C ABI field type: {c_name}')
                if not pointer:
                    for size in reversed(re.findall(r'\[(\d+)\]', dimensions or '')):
                        field_type = field_type * int(size)
                fields.append((field_name, field_type))
            result = type(resolved, (ctypes.Structure,), {'_fields_': fields})
            built[resolved] = result
            building.remove(resolved)
            return result

        entry_type = build(entry_name)
        for declared in inouts.values():
            element = declared.rsplit(' OF ', 1)[-1].strip()
            for name in bodies:
                if name.upper() in {element.upper(), element.upper() + '_TYPE'}:
                    build(name)
        targets = dict(inouts)
        for raw in bodies[entry_name].splitlines():
            field = engine.FIELD_RE.fullmatch(raw.strip())
            if field and '*' in field[1] and field[2] not in targets:
                targets[field[2]] = field[1].replace('*', '').strip()
        return entry_type, built, targets

    engine.parse_structs = parse_structs
