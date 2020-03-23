Welcome to pyGTM's documentation
================================


What is it?
-----------

pyGTM is a python code to calculate light propagation in multi-layered stacks of arbitrary (isotropic, uniaxial and biaxial) materials. This means that pyGTM handles general, 3x3 *complex permittivity tensors*. It is based on a generalized `transfer matrix method <https://en.wikipedia.org/wiki/Transfer-matrix_method_(optics)>`_ (GTM) [#f1]_ [#f2]_. In addition to describing the reflection and transmission of light at each interface, it can also compute the absorption in the material, thus enabling layer-resolved visualization of the light absorption [#f3]_. 

pyGTM initially spun off the Matlab code from `Nikolai Passler <https://pc.fhi-berlin.mpg.de/latdyn/peopledir/group-members/nikolai-pasler/>`_ 
and `Alexander Paarman <https://pc.fhi-berlin.mpg.de/latdyn/peopledir/group-members/alexander-paarmann/>`_ 
in the Fritz-Haber Institute `Lattice Dynamics group <https://pc.fhi-berlin.mpg.de/latdyn/>`_ in Berlin. 

The core of the module is derived from the work of `D. Dietze's <https://github.com/ddietze>`_ `FSRStools <https://github.com/ddietze/FSRStools>`_ which implemented the 4X4 transfer matrix formalism of Yeh [#f4]_ [#f5]_, which lead to singularities in some particular cases. 
This new formulation is thus more stable, and hopefully suited to study new optical phenomena with arbitrary material properties.


.. rubric:: References

.. [#f1] Passler, N. C. and Paarmann, A., *JOSA B* **34**, 2128 (2017) `10.1364/JOSAB.34.002128 <http://doi.org/10.1364/JOSAB.34.002128>`_
.. [#f2] Passler, N. C. and Paarmann, A., *JOSA B* **36**, 3246 (2019) `10.1364/JOSAB.36.003246 <http://doi.org/10.1364/JOSAB.36.003246>`_
.. [#f3] Passler, N. C., Jeannin, M. and Paarmann, A., *arXiv* (2020) `arxiv:2002.03832 <https://arxiv.org/abs/2002.03832>`_
.. [#f4] Yeh, P., *J. Opt. Soc. Am.*, **69**, 742 (1979) `10.1364/JOSA.69.000742 <https://doi.org/10.1364/JOSA.69.000742>`_
.. [#f5] Yeh, P., *Surf. Sci.*, **96**, 41 (1980) `10.1016/0039-6028(80)90293-9 <https://doi.org/10.1016/0039-6028(80)90293-9>`_ 


Download and installation
-------------------------
You can download the latest version on `github <https://github.com/pyMatJ/pyGTM>`_. There is no installation file to run, just make sure that the main GTM foled is in your `pythonpath`.



Documentation Contents
----------------------

.. toctree::
   :maxdepth: 2
   
   Tutorial
   GTMcore
   Permittivities
   

Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
