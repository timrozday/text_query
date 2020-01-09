from setuptools import setup

setup(
    # Needed to silence warnings (and to be a worthwhile package)
    name='text_query',
    url='https://github.com/timrozday/text_query',
    author='Tim Rozday',
    author_email='timrozday@ebi.ac.uk',
    # Needed to actually package something
    packages=['text_query'],
    # Needed for dependencies
    install_requires=[],
    version='0.1',
    # The license can be anything you like
    license='Do what you like with it (just nothing evil)',
    description='A few functions for parsing XML, creating varients of sentences and querying them with an index.',
    # We will also need a readme eventually (there will be a warning)
    long_description='A few functions for parsing XML, creating varients of sentences and querying them with an index.',
    # long_description=open('README.txt').read(),
)
