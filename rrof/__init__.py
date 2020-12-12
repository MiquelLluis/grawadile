"""OLD AD-HOC NOTES ABOUT 'gs1'

h: es el paso de tiempo de la derivada que se usa para calcular el
    gradiente. El hecho de que sea 1 es porque el método está pensado
    para imágenes con distancia entre pixels 1. La justificación
    matemática probablemente sea mas compleja. No puede ser 0.

beta: dado que el gradiente va dividiendo, la ecuación de ROF es
	singular cuando el gradiente vale 0. Para evitar esto se introduce
	un término beta que en principio debe ser pequeño para evitar este
	problema. Mis valores son puramente empíricos, sacados de hacer
	pruebas con varios valores con varios órdenes de magnitud de
	diferencia. Tampoco puede ser 0.

lambda: es el regularizador de Lagrange de la ecuación. Definido
	positivo. Balance los 2 términos de la ecuación, el término de
	mínimos cuadrados y el término de regularización del gradiente.
	No puede ser cero, puesto que entonces solo estaríamos aplicando
	una sola parte de la ecuación, o bien estarías haciendo un ajuste
	por mínimos cuadrados o bien solo estarías reduciendo el gradiente. 
"""


try:
	from .gs import gs1

except ImportError:
	from .gs1_compile import compile_gs
	compile_gs()
	del compile_gs
	
	from .gs import gs1