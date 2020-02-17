from numpy import f2py
with open("gs1.f") as sourcefile:
    sourcecode = sourcefile.read()
f2py.compile(sourcecode, modulename='gs',)