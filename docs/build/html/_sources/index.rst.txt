Welcome to pyGTM's documentation
================================


What is it?
-----------

pyGTM is a python code to calculate light propagation in multi-layered stacks of arbitrary (isotropic, uniaxial and biaxial) materials. This means that pyGTM handles general, 3x3 complex permittivity tensors. It is based on a generalized transfer matrix method (GTM). In addition to describing the reflection and transmission of light at each interface, it can also compute the absorption in the material, thus enabling layer-resolved visualization of the light absorption. 

pyGTM initially spun off the Matlab code from `Nikolai Passler <https://pc.fhi-berlin.mpg.de/latdyn/peopledir/group-members/nikolai-pasler/>`_ 
and `Alexander Paarman <https://pc.fhi-berlin.mpg.de/latdyn/peopledir/group-members/alexander-paarmann/>`_ 
in the Fritz-Haber Institute `Lattice Dynamics group <https://pc.fhi-berlin.mpg.de/latdyn/>`_ in Berlin. 

Download and installation
-------------------------
You can download the latest version on `github <https://github.com/pyMatJ/pyGTM>`_. There is no installation file to run, just make sure that the main GTM foled is in your `pythonpath`.



Documentation Contents
----------------------

.. toctree::
   :maxdepth: 2
   
   GTMcore
   Permittivities
   Examples

Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
