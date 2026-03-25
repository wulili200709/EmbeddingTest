import pathlib

LEGACY_DIR = pathlib.Path(__file__).resolve().parent
EMBEDDING_ROOT = LEGACY_DIR.parent
fp = EMBEDDING_ROOT / 'tool_page_pyside6.py'
src = fp.read_text(encoding='utf-8')
start_marker = '        # IO \u8c03\u8bd5\uff08\u5bf9\u8bdd\u6846\u7528\uff09'
end_marker = '        iol.setRowStretch(6, 1)'
start_idx = src.find(start_marker)
end_idx = src.find(end_marker)
if start_idx < 0 or end_idx < 0:
    raise SystemExit('ERROR markers %s %s' % (start_idx, end_idx))
end_idx += len(end_marker)
nb = (LEGACY_DIR / 'root_artifacts' / '_snippet_esc.txt').read_text(encoding='ascii')
new_block = nb.encode('ascii').decode('unicode_escape')
src = src[:start_idx] + new_block + src[end_idx:]
if src.count('size=(900, 420)') != 1:
    raise SystemExit('ERROR size count %s' % src.count('size=(900, 420)'))
src = src.replace('size=(900, 420)', 'size=(900, 480)', 1)
fp.write_text(src, encoding='utf-8')
print('OK', len(src))
