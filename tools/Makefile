#
## Makefile
# Masao Someki, 2019-01-30
#

.PHONY: all clean

all: venv infer


venv:
	python3 -m venv venv

infer:
	. venv/bin/activate; pip install -U pip; pip install -r requirements.txt

onnx_export:
	. venv/bin/activate; pip install -r requirements_export.txt

clean:
	rm -fr venv dist build *.egg-info

#vim:ft=make
