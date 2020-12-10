from pathlib import Path

from numpy import f2py


def compile_gs():
	path = Path(__file__).parent
	path_code = path / 'gs1.f'
	path_compiled = path / 'gs'

	with open(path_code, 'rb') as sourcefile:
	    sourcecode = sourcefile.read()
	f2py.compile(sourcecode, modulename=path_compiled)


if __name__ == '__main__':
	compile_gs()