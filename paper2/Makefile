.PHONY: all clean results figures

RESULTS ?= results_all.json.gz

all: main.pdf

# Exact proofs (standard library only).
proofs:
	cd proofs && python3 run_all.py

# Measured-results section from the passed records (requires the installed framework).
results:
	python3 reproduce/analysis.py $(RESULTS) --paper

# Figures from the run records.
figures:
	python3 generate_figures.py $(RESULTS)

main.pdf: main.tex
	tectonic main.tex

clean:
	rm -f main.aux main.log main.out main.pdf
