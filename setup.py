from setuptools import setup

setup(
   name='pyGTM',
   version='2.0.0',
   description='Generalized Transfer Matrix program',
   author='Mathieu Jeannin',
   author_email='math.jeannin@free.fr',
   packages=['pyGTM'],  #same as name
   install_requires=['numpy', 'scipy'], #external packages as dependencies
   scripts=[
            'GTM/GTMcors',
            'GTM/Permittivities',
           ]
)