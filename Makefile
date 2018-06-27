

all:
	python setup.py build

clean:
	python setup.py clean
	rm -rf build/
	rm -f *.pyc *~ */*~ */*.pyc *.aux *.log *.pdf greensconvolution/*~

commit: clean
	hg addrem
	hg commit
