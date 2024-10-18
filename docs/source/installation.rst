Installation
============

Dependencies
------------

:math:`\texttt{tropygal}` only requires `numpy <https://numpy.org/>`__ and `scipy <https://scipy.org/>`__ for the main entropy
estimators. To the specific task of estimating the density of
states for a DF depending on energy and angular momentum, it
currently requires `Agama <https://github.com/GalacticDynamics-Oxford/Agama>`__.

Installing
----------

:math:`\texttt{tropygal}` can be installed from the PyPI module typing::
  
  pip install --user tropygal

It can also be installed by downloading the source or cloning the
`GitHub repository <https://github.com/lberaldoesilva/tropygal>`__ and
typing::

  python setup.py install --user

from its root directory. In case you want it installed in a specific directory::
  
  python setup.py install --prefix=/WHERE/YOU/WANT/IT
