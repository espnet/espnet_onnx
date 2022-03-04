#
## Makefile
# Masao Someki, 2019-01-30
#

.PHONY: all clean

all: cpu gpu


cpu:
	python3 -m venv venv
		. venv/bin/activate; pip install -U pip; pip install -r requirements.txt

gpu:
	python3 -m venv venv
		. venv/bin/activate; pip install -U pip; pip install -r requirements_gpu.txt

clean:
	rm -fr venv dist build *.egg-info

#vim:ft=make
