.PHONY: paper results

paper: main.pdf

# Tables, figures and in-text numbers from the collected run records.
results:
	python3 generate_results.py results_all.json

main.pdf: main.tex
	tectonic main.tex
