.PHONY: all clean

all: main.pdf

main.pdf: main.tex main.bbl refs.bib figs/fig1_both_gaps.pdf figs/fig2_recall_50x.pdf figs/fig3_a5_seed_ablation.pdf
	tectonic main.tex

clean:
	rm -f main.aux main.log main.out main.blg main.pdf
