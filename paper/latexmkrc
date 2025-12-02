# Engine: pdflatex + bibtex (classic toolchain). Switch MINTED=1 to enable -shell-escape.
$pdf_mode = 1;

# Base commands
$pdflatex_cmd = 'pdflatex -interaction=nonstopmode -file-line-error %O %S';
$pdflatex_cmd_shell = 'pdflatex -shell-escape -interaction=nonstopmode -file-line-error %O %S';

# Toggle minted via env var
my $minted = $ENV{MINTED} // 0;
$latex = $minted ? $pdflatex_cmd_shell : $pdflatex_cmd;

# Tell latexmk which LaTeX to run in -pdf mode
$pdflatex = $latex;

# Use bibtex (NOT biber)
$bibtex = 'bibtex %O %B';

$max_repeat = 5;
$recorder = 1;

# Keep clean target civilized (preserve sources/PDF)
$clean_ext = join(' ',
  'aux bbl blg fdb_latexmk fls log run.xml synctex.gz toc out',
  'nav snm vrb lot lof lol loc lox idx ilg ind cut pgf-plot.gnuplot',
  'thm auxlock xdv dvi xml tmp.gz .ptc .tc .t1 .t2 .t3 .t4 .t5 .ttt',
  'acn acr alg glg glo gls ist lg loa lcs lmd lmr lms'
);

$pdf_previewer = 'open -a Skim %S';

$pdf_previewer_shell = 'open -a Skim %S';