# Mods for matplotlib
import matplotlib.ticker
from matplotlib import rcParams


# Set a fixed order of magnitude for the ScalarFormatter.
# From @burnpank at https://stackoverflow.com/questions/29827807/python-matplotlib-scientific-axis-formating
if 'axes.formatter.useoffset' in rcParams:
    # None triggers use of the rcParams value
    useoffsetdefault = None
else:
    # None would raise an exception
    useoffsetdefault = True

class FixedScalarFormatter(matplotlib.ticker.ScalarFormatter):
    def __init__(self, fformat, orderOfMagnitude=0, useOffset=useoffsetdefault,
                 useMathText=None, useLocale=None):
        super(FixedScalarFormatter,self).__init__(
            useOffset=useOffset, useMathText=useMathText, useLocale=useLocale
        )
        self.base_format = fformat
        self.orderOfMagnitude = orderOfMagnitude

    def _set_orderOfMagnitude(self, range):
        """ Set orderOfMagnitude to best describe the specified data range.

        Does nothing except from preventing the parent class to do something.
        """
        pass

    def _set_format(self, vmin, vmax):
        """ Calculates the most appropriate format string for the range (vmin, vmax).

        We're actually just using a fixed format string.
        """
        self.format = self.base_format
        if self._usetex:
            self.format = '$%s$' % self.format
        elif self._useMathText:
            self.format = '$\mathdefault{%s}$' % self.format   