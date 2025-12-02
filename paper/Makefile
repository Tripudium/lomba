MAIN=main
MINTED?=0

.PHONY: all draft final watch clean lint format bbl arxiv

all: draft

draft:
	@MINTED=$(MINTED) latexmk -pdf -interaction=nonstopmode $(MAIN).tex

final:
	@MINTED=$(MINTED) latexmk -pdf -interaction=nonstopmode -halt-on-error $(MAIN).tex

watch:
	@MINTED=$(MINTED) latexmk -pdf -pvc -interaction=nonstopmode $(MAIN).tex

bbl:
	@MINTED=$(MINTED) latexmk -pdf -interaction=nonstopmode $(MAIN).tex

clean:
	@latexmk -C
	@rm -f $(MAIN).bbl $(MAIN).run.xml

lint:
	@chktex -q -I0 -n1 -n8 -n46 $(MAIN).tex sections/*.tex || true

format:
	@latexindent -w $(MAIN).tex sections/*.tex

arxiv: clean
	@python3 - <<-'PY'
	import os, shutil, pathlib
	root = pathlib.Path('.')
	out = root/'arxiv'
	shutil.rmtree(out, ignore_errors=True)
	out.mkdir()
	keep_ext = {'.tex','.sty','.cls','.bst','.bib','.bbl','.bbx','.cbx','.png','.jpg','.pdf','.eps','.tikz'}
	for p in root.rglob('*'):
	    if p.is_dir(): continue
	    if p.parts[0] in {'.git','.github','arxiv'}: continue
	    if p.suffix.lower() in keep_ext:
	        tgt = out / p.relative_to(root)
	        tgt.parent.mkdir(parents=True, exist_ok=True)
	        shutil.copy2(p, tgt)
	print("arxiv package in ./arxiv")
	PY
