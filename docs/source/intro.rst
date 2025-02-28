Introduction
============

:math:`\texttt{tropygal}` is a pure-python package for estimating the
`differential entropy <https://en.wikipedia.org/wiki/Differential_entropy>`__
of a probability distribution function (pdf) :math:`f(\vec{w})` in
:math:`d`-dimensions, in the context of galactic dynamics.
The entropy is estimated with a Monte Carlo method,
i.e. using a sample of the underlying pdf,
without assuming any specific form for :math:`f(\vec{w})`.
The same estimators can also be used very broadly in other contexts.
Detailed expressions and numerical tests can be found in `Beraldo e Silva et al (2024)
<https://arxiv.org/abs/2407.07947>`__. Please cite this paper if you
use :math:`\texttt{tropygal}`.

Main expressions
----------------

A convenient entropy definition is [1]_

.. math:: S[f] \equiv - \int f(\vec{w}) \ln \left(\frac{f}{\mu}\right)
      \mathrm{d}^d\vec{w}
      :label: eq_S_def

where :math:`\mu` makes the argument of :math:`\ln()` dimensionless,
and the pdf is assumed to be normalized such that

.. math:: \int f(\vec{w}) \mathrm{d}^d\vec{w} = 1.
	  :label: eq_norm

Given a size-:math:`N` sample of
:math:`f(\vec{w})`, the entropy is estimated as

.. math:: \hat{S} = - \frac{1}{N}\sum_i^N \ln \left( \frac{\hat{f}_i}{\mu_i} \right),

where :math:`\hat{f}_i` is the estimate of :math:`f(\vec{w})` at
point/particle/star :math:`i`. In principle, any density estimate
method can be used for :math:`\hat{f}_i`, but certain methods are
ideal to precisely estimate the entropy - see e.g. `Hall &
Morton (1993) <https://rdcu.be/dXemG>`__, `Beirlant et al (2001)
<http://jimbeck.caltech.edu/summerlectures/references/Entropy%20estimation.pdf>`__.

:math:`\texttt{tropygal}` currently implements the :math:`k`-th Nearest Neighbor (kNN)
estimator, as detailed in e.g. `Leonenko, Pronzato & Savani (2008)
<https://projecteuclid.org/journals/annals-of-statistics/volume-36/issue-5/A-class-of-R%c3%a9nyi-information-estimators-for-multidimensional-densities/10.1214/07-AOS539.full>`__,
`Biau & Devroye (2015)
<https://link.springer.com/book/10.1007/978-3-319-25388-6>`__. It also
optionally implememnts a simple correction for the bias in the
estimates, as proposed by `Charzynska & Gambin (2015)
<https://www.mdpi.com/1099-4300/18/1/13>`__.
	  

Typically in galactic dynamics, :math:`d=6` and
:math:`\vec{w}=(\vec{r}, \vec{v})` are the phase-space coordinates
(and we simply refer to :math:`f(\vec{w})` as the DF). The quantity
:math:`\mu` in Eq. :eq:`eq_S_def` can also accommodate densities of
states in applications where the DF depends only on integrals of
motion (as implied by the Jeans' theorem for stationary samples).

Change to normalized coordinates
--------------------------------

Note that :math:`\vec{w}` involves coordinates of possibly different
units and very dissimilar magnitudes (e.g. positions and
velocities). To estimate the entropy, we calculate (Euclidean)
distances between sample points, so it's useful to change variables to
coordinates normalized by the respective dispersions, :math:`x' =
x/\sigma_x, y' = y/\sigma_y, z' = z/\sigma_z` and so on, allowing a
proper distance definition.

The DF of the new coordinates :math:`\vec{w}' =(\vec{r}', \vec{v}')`
is :math:`f'(\vec{w}') = |\Sigma|f(\vec{w})`, where
:math:`|\Sigma|=\sigma_x\cdot\sigma_y\dots\sigma_{v_z}`. Setting
:math:`\mu=|\Sigma|^{-1}` in Eq. :eq:`eq_S_def`, the entropy reduces
to

.. math:: S = - \int f(\vec{w}) \ln (|\Sigma|f) \mathrm{d}^6\vec{w} = - \int f'(\vec{w}') \ln f' \mathrm{d}^6\vec{w}',
      :label: eq_S_def_2

and can be better estimated from the sample with normalized coordinates as

.. math:: \hat{S} = - \frac{1}{N}\sum_i^N \ln \hat{f}_i',
	  
where :math:`\hat{f}_i'` is the estimate of :math:`f'(\vec{w}')` at
point/particle/star :math:`i`. After estimating :math:`S` using the
normalized coordinates, one might be interested in obtaining the
differential entropy as more commonly defined, i.e. :math:`- \int f
\ln f \mathrm{d}^6\vec{w}`. From Eq. :eq:`eq_S_def_2` and using Eq.
:eq:`eq_norm`, we have:

.. math:: - \int f(\vec{w}) \ln f \mathrm{d}^6\vec{w} \simeq \hat{S} + \ln |\Sigma|.


Code usage
----------

For a simple usage, let's start importing the relevant modules::

  >>> import numpy as np
  >>> import tropygal

Assume we have a sample of 6D coordinates :math:`\vec{w}=(x, y, z,
v_x, v_y,v_z)` for :math:`N` particles at a given time, so each of
these is an array of size :math:`N`. As discussed above, it's useful
to change variables to coordinates normalized by the respective
dispersions, :math:`x' = x/\sigma_x, y' = y/\sigma_y, z' = z/\sigma_z`
and so on::
  
  >>> data = np.column_stack((x, y, z, vx, vy, vz))
  >>> Sigma = np.array([0.5*(np.percentile(coord, 84) - np.percentile(coord, 16)) for coord in data.T])

Sigma is an array storing the typical dispersion of each of the 6D
coordinates - instead of standard deviations, we use percentiles,
which are more robust against outliers and are finite for any pdf.

The entropy is estimated as::

  >>> S = tropygal.entropy(data/Sigma)

Explicitly setting the optional arguments to their standard values::

  >>> S = tropygal.entropy(data/Sigma, mu=1, k=1, correct_bias=False, vol_correction='cube', l_cube_over_d=None, workers=-1)

In the last line, 'k' is the used neighbor (the k in kNN),
'correct_bias' sets whether the estimate should be corrected for the
bias as proposed by `Charzynska & Gambin (2015)
<https://www.mdpi.com/1099-4300/18/1/13>`__, 'vol_correction'
specifies details about the assumed support and the volume shape
around each point (currently only accepst 'cube'), 'l_cube_over_d' is
the side of the cube around each point :math:`i` divided by the
distance :math:`D_{ik}` to its k-th neighbor - the standard is the
diagonal of the cube inscribed to the sphere of radius :math:`D_{ik}`,
i.e. :math:`l_i = (2/\sqrt{d})D_{ik}`, and finally 'workers' is the
number of CPUs to be used in the nearest neighbor identification (-1
means all available).

Note that larger values for 'k' typically introduce larger biases and
smaller fluctuations in the entropy estimate. The bias correction
proposed by `Charzynska & Gambin (2015)
<https://www.mdpi.com/1099-4300/18/1/13>`__ seems to suppress the bias
without introducing extra noise - see also Fig. 5 in `Beraldo e Silva
et al (2024) <https://arxiv.org/abs/2407.07947>`__.

See the tutorials for more complete examples.
  
.. rubric:: Footnotes

.. [1] See specifically `https://en.wikipedia.org/wiki/Differential_entropy#Variants <https://en.wikipedia.org/wiki/Differential_entropy#Variants>`__.
