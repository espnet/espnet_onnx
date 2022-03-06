#
## Makefile
# Masao Someki, 2019-01-30
#

.PHONY: all clean

all: cpu gpu


cpu:
	python3 -m venv venv
		. venv/bin/activate; pip install -U pip; pip install -r requirements.txt

cpu_export:
	python3 -m venv venv
		. venv/bin/activate; pip install -U pip; pip install -r requirements.txt

gpu:
	. venv/bin/activate; pip install torch torchvision

clean:
	rm -fr venv dist build *.egg-info

#vim:ft=make
