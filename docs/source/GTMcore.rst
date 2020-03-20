.. default-domain:: py

.. _GTMcore-label:

GTMcore module
-------------------------------

.. automodule:: GTM.GTMcore

   The main algorithm is based on solving the eigenvalue equation for the electric field in each layer. The results are then propagated in a step-wised manner layer by layer accross the entire stack. 

   .. image:: Layers_Field_Buildup.png

   The Layer class
   ---------------
   .. autoclass:: Layer
      :members:


   The System class
   ----------------
   .. autoclass:: System
      :members:
