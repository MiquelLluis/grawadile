import glob
import pathlib
import shutil

from numpy import f2py


def compile_gs():
	path = pathlib.Path(__file__).parent
	path_code = path / 'gs1.f'
	module_name = 'gs'

	with open(path_code, 'rb') as f:
	    sourcecode = f.read()

	res = f2py.compile(sourcecode, modulename=module_name)
	if res != 0:
		raise ImportError("compilation of 'gs1.f' failed")

	gs_path = glob.glob("gs*.so")[0]
	gs_final_path = path / gs_path
	shutil.move(gs_path, gs_final_path)
