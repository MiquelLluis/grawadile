from numpy import f2py


def compile_gs():
	with open("gs1.f") as sourcefile:
	    sourcecode = sourcefile.read()

	f2py.compile(sourcecode, modulename='gs')


if __name__ == '__main__':
	compile_gs()